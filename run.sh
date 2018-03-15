cd build
make -j8
cd ..

./build/bin/dso_dataset files=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/图像公开数据集/kitti/kitti_gray/dataset/sequences/06 \
calib=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/图像公开数据集/kitti/calib/06.txt \
calibRight=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/图像公开数据集/kitti/calib/06.txt \
vocab=/home/ren/work/dso_semantic/vocab/ORBvoc.bin \
gtPath=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/图像公开数据集/kitti/dataset/poses/06.txt \
resultFile=/home/ren/work/fdso0117/fdso/result/result06.txt \
preset=2 \
openLoop=0 \
mode=1
