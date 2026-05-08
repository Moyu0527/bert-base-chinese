import os
import json
import config
from data_processing.weat_preprocessor import WEATPreprocessor
from data_processing.cdial_parser import CDialBiasParser


def main():
    print("=" * 60)
    print("Phase 1-3: WEAT and CDial-Bias Data Preprocessing")
    print("=" * 60)

    weat_dir = config.ensure_dir(config.DATA_CONFIG["weat_data_dir"])
    cdial_dir = config.ensure_dir(config.DATA_CONFIG["cdial_bias_data_dir"])

    print("\n[Step 1] WEAT Chinese Preprocessing...")
    weat_processor = WEATPreprocessor()

    print("Building WEAT Chinese term mappings...")
    weat_chinese = weat_processor.build_chinese_weat()

    weat_output_path = weat_processor.save_weat_data(weat_dir)

    print("\n[Step 2] CDial-Bias Dataset Parsing...")
    cdial_parser = CDialBiasParser(data_dir=cdial_dir)

    sample_data_path = os.path.join(cdial_dir, "cdial_bias_sample.json")
    with open(sample_data_path, "w", encoding="utf-8") as f:
        json.dump(cdial_parser._create_sample_cdial_data(), f, ensure_ascii=False, indent=2)

    raw_data = cdial_parser.load_raw_data(sample_data_path)
    print(f"Loaded {len(raw_data)} raw CDial-Bias samples")

    parsed_data = cdial_parser.parse()
    print(f"Parsed {len(parsed_data)} bias-related samples")

    cdial_output_path = cdial_parser.save_parsed_data(cdial_dir)

    cdial_parser.print_statistics()

    print("\n" + "=" * 60)
    print("Phase 1-3 completed!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  WEAT Chinese: {weat_output_path}")
    print(f"  CDial-Bias:    {cdial_output_path}")


if __name__ == "__main__":
    main()
