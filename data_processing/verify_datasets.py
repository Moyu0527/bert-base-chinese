import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from data_processing.weat_preprocessor import WEATPreprocessor
from data_processing.cdial_parser import CDialBiasParser
import config


def verify_weat_data():
    print("=" * 60)
    print("Verifying WEATHub Official Dataset")
    print("=" * 60)
    
    preprocessor = WEATPreprocessor()
    
    official_data = preprocessor.load_official_weat()
    
    if official_data:
        print(f"\n[WEAT] Number of tests: {len(official_data)}")
        
        for i, test in enumerate(official_data):
            print(f"\n[{i+1}] WEAT Test: {test.get('weat', '')}")
            print(f"       Language: {test.get('language', '')}")
            print(f"       Attr1: {test.get('attr1_category', '')} ({len(test.get('attr1_examples', []))} words)")
            print(f"       Attr2: {test.get('attr2_category', '')} ({len(test.get('attr2_examples', []))} words)")
            print(f"       Targ1: {test.get('targ1_category', '')} ({len(test.get('targ1_examples', []))} words)")
            print(f"       Targ2: {test.get('targ2_category', '')} ({len(test.get('targ2_examples', []))} words)")
            
            sample_attr1 = test.get('attr1_examples', [])[:3]
            sample_attr2 = test.get('attr2_examples', [])[:3]
            sample_targ1 = test.get('targ1_examples', [])[:3]
            sample_targ2 = test.get('targ2_examples', [])[:3]
            
            print(f"       Sample attr1: {', '.join(sample_attr1)}...")
            print(f"       Sample attr2: {', '.join(sample_attr2)}...")
            print(f"       Sample targ1: {', '.join(sample_targ1)}...")
            print(f"       Sample targ2: {', '.join(sample_targ2)}...")
    
    return official_data


def verify_cdial_bias_data():
    print("\n" + "=" * 60)
    print("Verifying CDial-Bias Official Dataset")
    print("=" * 60)
    
    parser = CDialBiasParser()
    
    official_data = parser.load_official_cdial_bias()
    
    if official_data:
        print(f"\n[CDial-Bias] Description: {official_data.get('description', '')}")
        print(f"[CDial-Bias] Paper: {official_data.get('paper', '')}")
        print(f"[CDial-Bias] Source: {official_data.get('source', '')}")
        
        stats = official_data.get('statistics', {})
        print(f"\n[CDial-Bias] Statistics:")
        print(f"       Total samples: {stats.get('total_samples', 0)}")
        
        by_topic = stats.get('by_topic', {})
        if by_topic:
            print(f"       By topic:")
            for topic, count in by_topic.items():
                print(f"         {topic}: {count}")
        
        by_attitude = stats.get('by_attitude', {})
        if by_attitude:
            print(f"       By attitude:")
            attitude_map = {0: "anti_bias", 1: "neutral", 2: "biased", 3: "irrelevant"}
            for attitude, count in by_attitude.items():
                label = attitude_map.get(attitude, f"unknown_{attitude}")
                print(f"         {label}: {count}")
        
        data_list = official_data.get('data', [])[:5]
        print(f"\n[CDial-Bias] Sample data (first 5):")
        for i, item in enumerate(data_list):
            print(f"\n       [{i+1}] Topic: {item.get('topic', '')}")
            print(f"         Q: {item.get('q', '')}")
            print(f"         A: {item.get('a', '')}")
            print(f"         Group: {item.get('group', '')}")
            print(f"         Attitude: {item.get('attitude', '')}")
    
    return official_data


def main():
    print("\n" + "=" * 70)
    print("Official Dataset Import Verification and Annotation")
    print("=" * 70)
    
    weat_data = verify_weat_data()
    cdial_data = verify_cdial_bias_data()
    
    print("\n" + "=" * 70)
    print("Dataset Verification Complete!")
    print("=" * 70)
    
    if weat_data and cdial_data:
        print("\n✅ Both WEATHub and CDial-Bias official datasets are successfully imported!")
        print(f"   - WEATHub: {len(weat_data)} tests loaded")
        print(f"   - CDial-Bias: {len(cdial_data.get('data', []))} samples loaded")
        
        print("\n📊 Data Annotation Summary:")
        print("   - WEATHub: 6 WEAT tests (WEAT1, WEAT2, WEAT6, WEAT7, WEAT8, WEAT9)")
        print("   - WEATHub: Chinese translations for Flowers/Insects, Instruments/Weapons, Career/Family, etc.")
        print("   - CDial-Bias: 4 bias topics (gender, region, occupation, race)")
        print("   - CDial-Bias: Attitude labels (0=anti_bias, 1=neutral, 2=biased, 3=irrelevant)")
        print("   - CDial-Bias: Context-aware dialogue samples")
    
    elif weat_data:
        print("\n⚠️ Only WEATHub dataset loaded successfully")
    
    elif cdial_data:
        print("\n⚠️ Only CDial-Bias dataset loaded successfully")
    
    else:
        print("\n❌ Failed to load official datasets")


if __name__ == "__main__":
    main()
