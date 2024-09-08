seed=2023
max_epochs=20
learning_rate=1e-4
top_k=5
lambda_entity=0.6
lambda_predicate=0.3
lambda_sentence=1.0
lambda_soft=0.5
max_objects=6
weight_decay=1e-4

python -u main_test.py --dataset_name MSVD --entity_encoder_layer 2 --entity_decoder_layer 2 --max_objects 6 \
			--backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 \
			--backbone_3d_name C3D --backbone_3d_dim 2048 \
			--object_name vg_objects --object_dim 2048 --device cuda:0 \
			--max_epochs ${max_epochs} --save_checkpoints_every 500 --seed ${seed} \
			--data_dir ./data --model_name EDS --top_k ${top_k} \
			--learning_rate ${learning_rate} --lambda_entity ${lambda_entity} --lambda_predicate ${lambda_predicate} \
			--lambda_sentence ${lambda_sentence} --lambda_soft ${lambda_soft} --beam_alpha 0.5 --weight_decay ${weight_decay} \
			--msg "_${seed}_use_SSG_SSE_ISE" --use_SSG --use_SSE --use_ISE

