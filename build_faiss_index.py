#!/usr/bin/env python3
"""
سكريبت لبناء FAISS index من الملفات الموجودة
"""

import sys
import os
from services.vectorization.embedding_vectorizer import (
    build_faiss_index_from_existing, 
    check_faiss_index_exists
)

def build_faiss_for_dataset(dataset_name):
    """
    بناء FAISS index لداتا ست محدد
    """
    print(f"🔍 فحص الداتا ست: {dataset_name}")
    
    # التحقق من وجود FAISS index
    if check_faiss_index_exists(dataset_name):
        print(f"✅ FAISS index موجود بالفعل للداتا ست: {dataset_name}")
        return True
    
    # بناء FAISS index
    print(f"⏳ بناء FAISS index للداتا ست: {dataset_name}")
    try:
        faiss_path = build_faiss_index_from_existing(dataset_name)
        print(f"✅ تم بناء FAISS index بنجاح: {faiss_path}")
        return True
    except Exception as e:
        print(f"❌ فشل في بناء FAISS index لـ {dataset_name}: {e}")
        return False

def build_faiss_for_all_datasets():
    """
    بناء FAISS index لجميع الداتا ست المتوفرة
    """
    # قائمة الداتا ست المتوقعة
    datasets = [
        "antique/test",
        "beir_arguana", 
        "beir_trec-covid",
        "msmarco",
        "robust04"
    ]
    
    print("🚀 بدء بناء FAISS index لجميع الداتا ست...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(datasets)
    
    for dataset in datasets:
        print(f"\n📊 معالجة: {dataset}")
        if build_faiss_for_dataset(dataset):
            success_count += 1
        print("-" * 30)
    
    print(f"\n📈 النتائج النهائية:")
    print(f"✅ نجح: {success_count}/{total_count}")
    print(f"❌ فشل: {total_count - success_count}/{total_count}")

def main():
    """
    الدالة الرئيسية
    """
    print("🔧 سكريبت بناء FAISS index")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # بناء FAISS index لداتا ست محدد
        dataset_name = sys.argv[1]
        print(f"🎯 بناء FAISS index للداتا ست: {dataset_name}")
        build_faiss_for_dataset(dataset_name)
    else:
        # بناء FAISS index لجميع الداتا ست
        print("🌐 بناء FAISS index لجميع الداتا ست المتوفرة")
        build_faiss_for_all_datasets()

if __name__ == "__main__":
    main() 