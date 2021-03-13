python main.py \
	--exp ./expts/_temp\
	--gpu 1 \
	--load_model \
	--dataset shapenet_train \
	--_3d_loss_type init_model \
	--categ car \
	--loss bce\
	--affinity_loss \
	--optimise_pose \
	--lambda_ae 100. \
	--lambda_ae_mask 100. \
	--lambda_mask_fwd 1e-4 \
	--lambda_mask_bwd 1e-4 \
	--lambda_3d 10000. \
	--lambda_ae_pose 1. \
	--lambda_mask_pose 1. \
	--lambda_pose 1. \
	--lr 5e-4 \
	--batch_size 2 \
	--N_PROJ 4 \
	--save_n 500 \
	--save_model_n 2000 \
	--N_ITERS 200001 \
	--print_n 100
