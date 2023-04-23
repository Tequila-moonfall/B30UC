%==========================================================================
%  函数功能: 显示图像块的字典-显示原子用NxM的网格
%  输入参数：D - 字典
%           sz - 图像块的大小
%           n,m - 网格区域的大小
%           varargin -其它参数 
%                    - lines：用黑线隔开字典原子
%                    - whitelines：用白线隔开字典原子
%                    - highcontrast：增加显示的对比度
%  输出参数：x - 返回字典图像的位矩阵
%==========================================================================
function x = showdict(D,sz,n,m,varargin)

% -------------------------- 检查输入参数 ----------------------------------
if (size(D,2) < n*m)
  D = [D zeros(size(D,1),n*m-size(D,2))];
end

linewidth = 1;
highcontrast = 0;
drawlines = 0;
linecolor = 0;

for i = 1:length(varargin)
  if (~ischar(varargin{i}))
    continue;
  end
  switch(varargin{i})
    case 'highcontrast'
      highcontrast = 1;
    case 'lines'
      drawlines = 1;
    case 'whitelines'
      drawlines = 1;
      linecolor = 1;
    case 'linewidth'
      linewidth = varargin{i+1};
  end
end

% ------------------------ 生成字典图像 ------------------------------------
if (drawlines)
  
  D = [D ; nan(sz(1)*linewidth,size(D,2))];
  sz(2) = sz(2)+linewidth;
  x = col2im(D(:,1:n*m),sz,[n m].*sz,'distinct');
  sz = [sz(2) sz(1)];
  D = im2col(x',sz,'distinct');
  D = [D ; nan(sz(1)*linewidth,size(D,2))];
  sz(2) = sz(2)+linewidth;
  x = col2im(D(:,1:n*m),sz,[m n].*sz,'distinct');
  x = x';
  x = x(1:end-linewidth,1:end-linewidth);
  
  if (highcontrast)
    for i = 0:n-1
      for j = 0:m-1
        x(i*sz(1)+1:i*sz(1)+sz(1)-linewidth, j*sz(2)+1:j*sz(2)+sz(2)-linewidth) = ...
          imnormalize(x(i*sz(1)+1:i*sz(1)+sz(1)-linewidth, j*sz(2)+1:j*sz(2)+sz(2)-linewidth));
      end
    end
  else
    x = imnormalize(x);
  end
  
  x(isnan(x)) = linecolor;
  
else
  
  x = col2im(D(:,1:n*m),sz,[n m].*sz,'distinct');
  
  if (highcontrast)
    for i = 0:n-1
      for j = 0:m-1
        x(i*sz(1)+1:i*sz(1)+sz(1), j*sz(2)+1:j*sz(2)+sz(2)) = ...
          imnormalize(x(i*sz(1)+1:i*sz(1)+sz(1), j*sz(2)+1:j*sz(2)+sz(2)));
      end
    end
  else
    x = imnormalize(x);
  end
end


if (nargout==0)
  imshow(x);
end
