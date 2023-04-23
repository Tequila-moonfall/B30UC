%==========================================================================
%  函数功能: 根据给定的字典对信号进行稀疏编码。注意：表示每个信号的最大原子个数固定。
%  输入参数：D - 字典
%           X - 需要被表示的信号
%           L - 表示每个信号的最大原子个数
%  输出参数：A - 稀疏表示系数
%==========================================================================
function [A] = OMP(D,X,L); 

[n,P] = size(X);
[n,K] = size(D);
for k = 1:1:P,          % 按列来运算
    a = [];
    x = X(:,k);
    residual = x;       % 初始化残余量
    indx = zeros(L,1);  % 开辟空间
    for j=1:1:L,
        proj = D'*residual; % 求内积
        [maxVal,pos] = max(abs(proj));  % 找出内积最大的位置
        pos = pos(1);
        indx(j) = pos;                  % 保存位置信息
        a = pinv(D(:,indx(1:j)))*x;
        residual = x-D(:,indx(1:j))*a;  % 更新残余量
        if sum(residual.^2) < 1e-6      % 达到指定误差，退出本次迭代
            break;
        end
    end;
    temp = zeros(K,1);
    temp(indx(1:j)) = a;
    A(:,k) = sparse(temp);
end;
return;
