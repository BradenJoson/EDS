seed=2023
max_epochs=20
learning_rate=1e-4
top_k=3
lambda_entity=0.6
lambda_predicate=0.3
lambda_sentence=1.0
lambda_soft=0.5
max_objects=6

python -u main.py --dataset_name MSRVTT --entity_encoder_layer 3 --entity_decoder_layer 3 --max_objects ${max_objects} \
			--backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 \
			--backbone_3d_name C3D --backbone_3d_dim 2048 \
			--object_name vg_objects --object_dim 2048 --device cuda:0 \
			--max_epochs ${max_epochs} --save_checkpoints_every 500 --seed ${seed} \
			--data_dir ./data --model_name EDS --top_k ${top_k} \
			--learning_rate ${learning_rate} --lambda_entity ${lambda_entity} --lambda_predicate ${lambda_predicate} \
			--lambda_sentence ${lambda_sentence} --lambda_soft ${lambda_soft} --beam_alpha 0.8 --weight_decay 0 \
			--msg "_C3D_entity_${lambda_entity}_predicate_${lambda_predicate}_sentence_${lambda_sentence}_seed_${seed}" --use_SSG --use_SSE --use_ISE
