pwd
/usr/local/cuda-8.0/bin/nvcc -std=c++11 -c -o chamfer_utils/tf_nndistance_g.cu.o chamfer_utils/tf_nndistance_g.cu -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 chamfer_utils/tf_nndistance.cpp chamfer_utils/tf_nndistance_g.cu.o -o chamfer_utils/tf_nndistance_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -L /usr/local/cuda-8.0/lib64 -O2