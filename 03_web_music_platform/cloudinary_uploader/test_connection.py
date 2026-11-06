import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cloudinary
import cloudinary.api
from config import CLOUD_NAME, API_KEY, API_SECRET

def check_connection():
    print("=" * 60)
    print("KIỂM TRA KẾT NỐI TÀI KHOẢN CLOUDINARY")
    print("=" * 60)
    print(f"Cloud Name : {CLOUD_NAME}")
    print(f"API Key    : {API_KEY[:6]}...{API_KEY[-4:]}")
    
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )
    
    try:
        ping = cloudinary.api.ping()
        print(f"[OK] Ping Status : {ping.get('status')}")
    except Exception as e:
        print(f"[FAIL] Không thể kết nối tới Cloudinary: {e}")
        return False
        
    try:
        usage = cloudinary.api.usage()
        credits_info = usage.get("credits", {})
        storage_info = usage.get("storage", {})
        bandwidth_info = usage.get("bandwidth", {})
        
        print("\n[THÔNG TIN DUNG LƯỢNG & TÀI KHOẢN]")
        print(f"Plan Name        : {usage.get('plan', 'Free')}")
        print(f"Credits Used     : {credits_info.get('usage', 0.0)} / {credits_info.get('limit', 25.0)} ({credits_info.get('used_percent', 0.0):.1f}%)")
        print(f"Storage Used     : {storage_info.get('usage', 0) / (1024*1024):.2f} MB")
        print(f"Bandwidth Used   : {bandwidth_info.get('usage', 0) / (1024*1024):.2f} MB")
        print("\n=> Tài khoản hoàn toàn sẵn sàng để upload kho bài hát!")
        return True
    except Exception as e:
        print(f"[WARNING] Không thể lấy thông tin chi tiết gói: {e}")
        return True

if __name__ == "__main__":
    success = check_connection()
    sys.exit(0 if success else 1)
