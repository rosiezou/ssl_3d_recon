nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include
extra_tf = /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public
tf_only = /usr/local/lib/python2.7/dist-packages/tensorflow
cuda_include = /usr/local/cuda-8.0/include

all: chamfer_utils/tf_nndistance_so.so chamfer_utils/render_balls_so.so chamfer_utils/tf_auctionmatch_so.so
.PHONY : all

chamfer_utils/tf_nndistance_so.so: chamfer_utils/tf_nndistance.cpp chamfer_utils/tf_nndistance_g.cu.o
	g++ -std=c++11 chamfer_utils/tf_nndistance.cpp chamfer_utils/tf_nndistance_g.cu.o -o chamfer_utils/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -I $(cuda_include) -lcudart -L $(cudalib) -L $(tf_only) -ltensorflow_framework -O2

chamfer_utils/tf_nndistance_g.cu.o: chamfer_utils/tf_nndistance_g.cu
	$(nvcc) -std=c++11 -c -o chamfer_utils/tf_nndistance_g.cu.o chamfer_utils/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

chamfer_utils/render_balls_so.so: chamfer_utils/render_balls_so.cpp
	g++ -std=c++11 chamfer_utils/render_balls_so.cpp -o chamfer_utils/render_balls_so.so -shared -fPIC -O2

chamfer_utils/tf_auctionmatch_so.so: chamfer_utils/tf_auctionmatch.cpp chamfer_utils/tf_auctionmatch_g.cu.o 
	g++ -std=c++11 chamfer_utils/tf_auctionmatch.cpp chamfer_utils/tf_auctionmatch_g.cu.o -o chamfer_utils/tf_auctionmatch_so.so -shared -fPIC -I $(tensorflow) -I $(cuda_include) -lcudart -L $(cudalib) -L $(tf_only) -ltensorflow_framework -O2	

chamfer_utils/tf_auctionmatch_g.cu.o: chamfer_utils/tf_auctionmatch_g.cu
	$(nvcc) -std=c++11 -c -o chamfer_utils/tf_auctionmatch_g.cu.o chamfer_utils/tf_auctionmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -arch=sm_30

clean: rm -rf chamfer_utils/*.o chamfer_utils/*.so