%==========================================================================
%  函数功能: 根据给定的字典对信号进行稀疏编码，达到指定误差范围停止迭代
%  输入参数：D - 字典
%           X - 需要被表示的信号
%           errorGoal - 最大允许表示误差
%  输出参数：A - 稀疏表示系数
%==========================================================================
function [A] = OMPerr(D,X,errorGoal) 

[n,P] = size(X);
[n,K] = size(D);
E2 = errorGoal^2*n;
maxNumCoef = n/2;
A = sparse(size(D,2),size(X,2));
for k = 1:1:P,
    a =[];
    x = X(:,k);
    residual = x;
	indx = [];
	a = [];
	currResNorm2 = sum(residual.^2);
	j = 0;
    while currResNorm2>E2 && j < maxNumCoef,
		j = j+1;
        proj = D'*residual;                     % 求内积
        pos = find(abs(proj)==max(abs(proj)));
        pos = pos(1);
        indx(j) = pos;
        a = pinv(D(:,indx(1:j)))*x;
        residual = x-D(:,indx(1:j))*a;          % 更新残余量
		currResNorm2 = sum(residual.^2);
   end;
   if (~isempty(indx))
       A(indx,k)=a;
   end
end;
return;
