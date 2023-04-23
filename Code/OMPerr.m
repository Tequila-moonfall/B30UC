%==========================================================================
%  ��������: ���ݸ������ֵ���źŽ���ϡ����룬�ﵽָ����Χֹͣ����
%  ���������D - �ֵ�
%           X - ��Ҫ����ʾ���ź�
%           errorGoal - ��������ʾ���
%  ���������A - ϡ���ʾϵ��
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
        proj = D'*residual;                     % ���ڻ�
        pos = find(abs(proj)==max(abs(proj)));
        pos = pos(1);
        indx(j) = pos;
        a = pinv(D(:,indx(1:j)))*x;
        residual = x-D(:,indx(1:j))*a;          % ���²�����
		currResNorm2 = sum(residual.^2);
   end;
   if (~isempty(indx))
       A(indx,k)=a;
   end
end;
return;
