# LLaVA-format data

Place LLaVA-format train/test JSON files here. Generate them with the converter:

  cd ../llava && python convert_pcqa_to_llava_format.py \
    --descriptions_csv path/to/<annotated file> \
    --test_split_csv path/to/sjtu_data_info/<test_split_file> \
    --root_dir ../dataset --out_dir .. \
    --projections_subdir projections \
    --ply_column Ply_name --description_column Generated_Description --test_ply_column name

The shell scripts expect by default `train_pcqa_llava.json` and `test_pcqa_llava.json`. Override with script arguments or env vars if needed.
