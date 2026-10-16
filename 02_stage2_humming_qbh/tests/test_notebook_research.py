import ast
import json
import random
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from torch.autograd import Function
from typeguard import check_argument_types

ROOT = Path(__file__).resolve().parents[1] / 'notebooks' / 'kaggle' / 'archive'

def definitions(filename, markers):
    doc=json.loads((ROOT/filename).read_text(encoding='utf-8'))
    namespace=dict(globals())
    for cell in doc['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and any(m in source for m in markers):
            tree=ast.parse(source)
            tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
            exec(compile(tree,filename,'exec'),namespace)
    return namespace

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns=definitions('conformer_bytehum_qbh_kaggle.ipynb',[
            'class Model(BasicModel)', 'class SpectrogramTempoAndMaskDistortion',
            'class CrossBatchMemoryBank', 'class GradientReversalFunction',
            'class LocalDomainClassifier','def build_coverhunter_model','class LocalAlignmentClassifier',
            'class AudioPreprocessor','def _vectorized_local_maxmean','class PyCqt'])
        cls.ev=definitions('conformer_bytehum_qbh_evaluation_kaggle.ipynb',[
            'def fit_open_set_threshold','def build_retrieval_index'])

    def test_compile_and_no_source_dependency(self):
        for path in ROOT.glob('*kaggle.ipynb'):
            doc=json.loads(path.read_text(encoding='utf-8'))
            for c in doc['cells']:
                if c['cell_type']=='code':
                    source=''.join(c['source']); compile(source,str(path),'exec')
                    self.assertNotIn('from src.',source)

    def test_grid_identity_and_nonlinearity(self):
        Grid=self.ns['SpectrogramTempoAndMaskDistortion']
        x=torch.arange(96*83,dtype=torch.float32).reshape(1,96,83)
        identity=Grid((1.,1.),prob=1)(x)
        self.assertTrue(torch.allclose(x,identity,atol=.002))
        warped=Grid(prob=1)(x)
        self.assertEqual(warped.shape,x.shape)
        self.assertTrue(torch.isfinite(warped).all())
        self.assertFalse(torch.allclose(warped,x))
        axis=Grid(prob=1)._axis(83,1,x.device,x.dtype)[0]
        self.assertTrue(torch.all(axis[1:]>axis[:-1]))
        self.assertGreater(float((axis[1:]-axis[:-1]).std()),0)

        frequency_bands=torch.arange(96,dtype=torch.float32).view(1,96,1).expand(1,96,83)
        time_warped=Grid(prob=1)(frequency_bands)
        self.assertTrue(torch.allclose(time_warped,frequency_bands,atol=.002))
        mask_input=torch.linspace(-3,2,x.numel()).reshape_as(x)
        masked=Grid((1.,1.),max_mask_t=8,max_mask_f=4,prob=1)(mask_input)
        self.assertGreater(int((masked==mask_input.min()).sum()),1)

    def test_audio_chunking_does_not_append_redundant_tail(self):
        Preprocessor=self.ns['AudioPreprocessor']
        prep=Preprocessor(sample_rate=1,chunk_duration=10,hop_duration=3)
        audio=np.arange(20,dtype=np.float32)
        chunks=prep.slice_into_chunks(audio)
        self.assertEqual(len(chunks),4)
        np.testing.assert_array_equal(chunks[-1],audio[9:19])

    def test_vectorized_maxmean_matches_song_loop(self):
        score=self.ns['_vectorized_local_maxmean']
        query=F.normalize(torch.randn(7,8),dim=-1)
        lengths=torch.tensor([5,3,4])
        songs=[F.normalize(torch.randn(int(length),8),dim=-1) for length in lengths]
        padded=nn.utils.rnn.pad_sequence(songs,batch_first=True)
        mask=torch.arange(padded.shape[1])[None,:] < lengths[:,None]
        actual=score(query,padded,mask,song_batch_size=2)
        expected=torch.stack([(query @ song.T).max(dim=1).values.mean() for song in songs])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))

    def test_global_chunk_score_uses_max_cosine(self):
        score=self.ns['_vectorized_global_chunk_scores']
        query=F.normalize(torch.randn(3,8),dim=-1)
        lengths=torch.tensor([4,2,3])
        songs=[F.normalize(torch.randn(int(length),8),dim=-1) for length in lengths]
        padded=nn.utils.rnn.pad_sequence(songs,batch_first=True)
        mask=torch.arange(padded.shape[1])[None,:] < lengths[:,None]
        actual=score(query,padded,mask)
        expected=torch.stack([(query @ song.T).max() for song in songs])
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))

    def test_average_tie_rank_is_not_optimistic(self):
        rank,ties=self.ns['_average_tie_rank'](torch.tensor([.9,.9,.4]),1)
        self.assertEqual(ties,2)
        self.assertEqual(rank,1.5)

    def test_silent_cqt_maps_to_floor(self):
        PyCqt=self.ns['PyCqt']
        extractor=PyCqt.__new__(PyCqt)
        extractor._hop_size=.04
        extractor._sample_rate=16000
        extractor._kernel=None
        extractor._compute_cqt_spec=lambda *args:np.zeros((96,10),dtype=np.float32)
        result=extractor.compute_cqt(np.zeros(16000,dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result==-80.0))

    def test_missing_stage1_checkpoint_is_fatal(self):
        with self.assertRaises(FileNotFoundError):
            self.ns['build_coverhunter_model'](None)

    def test_prototype_diversity_penalizes_collapse(self):
        diversity=self.ns['prototype_diversity_loss']
        collapsed=torch.ones(2,4,8)
        orthogonal=torch.eye(8)[:4].unsqueeze(0).repeat(2,1,1)
        self.assertGreater(diversity(collapsed).item(),diversity(orthogonal).item())

    def test_lat_order_and_backward(self):
        loss=self.ns['local_alignment_triplet']
        a=torch.eye(4)[None].repeat(2,1,1).requires_grad_()
        ids=torch.tensor([0,0]); negatives=torch.tensor([1,1])
        good=loss(a,a,-a,ids,negatives)
        bad=loss(a,-a,a,ids,negatives)
        self.assertLess(good.item(),bad.item())
        bad.backward(); self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):loss(a,a,a,ids,ids)

    def test_projection_sizes(self):
        Model=self.ns['ConformerByteHumModel']
        for dim in (128,256,512):
            model=Model(num_classes=3,embedding_dim=dim,allow_random_init=True).cpu().eval()
            with torch.no_grad():
                outputs=model(torch.randn(2,1,96,83))
            self.assertEqual(outputs[0].shape,(2,dim))
            self.assertEqual(outputs[1].shape[-1],dim)
            self.assertEqual(outputs[2].shape,(2,3))
            clone=Model(num_classes=3,embedding_dim=dim,allow_random_init=True).cpu()
            clone.load_state_dict(model.state_dict(),strict=True)
            model.train()
            self.assertFalse(model.feature_extractor._global_cmvn.training)
            with torch.no_grad():
                outputs=model(torch.randn(2,1,96,83),compute_lac=False,compute_domain=False)
            self.assertIsNone(outputs[2])
            self.assertIsNone(outputs[3])

    def test_calibration_ties(self):
        fit=self.ev['fit_open_set_threshold']
        result=fit([.8,.9],[.1,.2,.8],0)
        self.assertGreater(result['threshold'],.8)
        self.assertEqual(result['empirical_far'],0)
        with self.assertRaises(ValueError):fit([.8],[],.01)

    def test_index_backends(self):
        build=self.ev['build_retrieval_index']
        rng=np.random.default_rng(1)
        for dim in (128,256,512):
            x=rng.normal(size=(100,dim)).astype('float32')
            faiss.normalize_L2(x)
            for kind in ('flat','hnsw','ivf'):
                index=build(x.copy(),kind,nlist=2,nprobe=2)
                _,ids=index.search(x[:5],1)
                np.testing.assert_array_equal(ids[:,0],np.arange(5))

if __name__=='__main__':unittest.main()
