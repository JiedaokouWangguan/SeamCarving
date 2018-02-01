# seamCarving

## todo:
  1. The order of the inserting/removing seams.(when inserting seams, the cost will decrease but not increase, so it will keep inserting seams)论文中的方法是，先假设只进行删除seam的操作，然后求是先删除row还是先删除col。但是首先这种方法不能推广到添加seam，因为添加seam时一次要选取多个seam然后一起复制。而删除的时候，论文中要求一次删除一个。第二，这种方法耗时巨大，因为在进行dp运算的时候，每次的删除都会改变整个图像，这样就要求重新计算dp的信息。
  
  
  2. The energy functions.
  3. How to get a good rate or a good function to calculate rate.
 
  
