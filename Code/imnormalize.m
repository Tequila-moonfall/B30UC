%==========================================================================
%  函数功能: 对图像的值进行归一化，对图像灰度值进行线性变换到[0,1]范围
%  输入参数：x - 输入信号
%  输出参数：A - 稀疏表示系数
%==========================================================================
function y = imnormalize(x)

maxval = max(x(:));
minval = min(x(:));

y = (x-minval) / (maxval-minval);
