import os
import sys
import win32com.client

def export_docx_to_pdf(docx_path, pdf_path):
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)
    
    print(f"Opening: {abs_docx}")
    print(f"Target PDF: {abs_pdf}")
    
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0  # wdAlertsNone
    
    doc = None
    try:
        # Open in read-only mode so it works even if user has it open
        doc = word.Documents.Open(abs_docx, ReadOnly=True)
        # 17 is wdFormatPDF
        doc.SaveAs(abs_pdf, FileFormat=17)
        print(f"Successfully converted to PDF: {abs_pdf}")
        print(f"PDF size: {os.path.getsize(abs_pdf)} bytes")
    except Exception as e:
        print(f"Error during PDF conversion: {e}", file=sys.stderr)
        raise
    finally:
        if doc:
            doc.Close(SaveChanges=False)
        # Do not call word.Quit() if user already had Word running with other docs,
        # but if we dispatched a background instance, word.Quit() closes only this instance or if word.Documents.Count == 0
        try:
            if word.Documents.Count == 0:
                word.Quit()
        except:
            pass

if __name__ == '__main__':
    export_docx_to_pdf("docs/report/BAO_CAO_PBL6_QBH_FINAL.docx", "docs/report/BAO_CAO_PBL6_QBH_FINAL.pdf")

