%==========================================================================
%  函数功能: 用K-SVD算法训练字典
%  输入参数：Data  - 包含N个n维信号的大小为nxN的矩阵
%           param - 其它参数
%                 - K：需要训练的字典元素个数
%                 - numIteration：迭代的次数
%                 - errorFlag：等于0则表示每个信号的稀疏系数个数固定，需要配置参数L；
%                              否则表示信号的稀疏系数不固定，需要配置参数errorGoal。
%                 - preserveDCAtom：等于1则字典的第一个原子为常量。
%                 - InitializationMethod：“DataElements”用信号本身作为字典
%                                         “GivenMatrix”用给定的矩阵作为初始字典
%                 - initialDictionary：初始化字典
%  输出参数：Dictionary - 训练得到的字典，大小为nxparam.K
%           output     - 输出结构体
%                      - CoefMatrix:稀疏矩阵
%                      - numCoef：平均表示每个原子的系数个数
%==========================================================================

function [Dictionary,output] = KSVD(    ...
                                        Data,... % 包含N个n维信号的大小为nxN的矩阵
                                        param)   % 其余参数

if (isfield(param,'errorFlag')==0)      % 表示每个信号的稀疏系数个数固定
    param.errorFlag = 0;
end

if (param.preserveDCAtom>0)             % 参数说明里面要求此变量等于1才执行以下操作，实际在这里变量大于0便执行操作了
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));% 填充第一列为常量，为了更好的表示自然图像
else
    FixedDictionaryElement = [];% 开辟空间
end

if (size(Data,2) < param.K)     % 若参数K大于信号的个数，则将数据集作为字典集 
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements')) 
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);% 将数据集的1到param.K-param.preserveDCAtom数据作为字典集  
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))   % 
    Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,1:param.K-param.preserveDCAtom);% 将initialDictionary的1到param.K-param.preserveDCAtom列作为字典集
end

if (param.preserveDCAtom)   % 字典第一个原子为常量
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end
% 归一化字典
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary))); %归一化
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.% 将字典集中的每个元素的化为正数
totalErr = zeros(1,param.numIteration);

%================================= 开始K-SVD算法 ===========================
for iterNum = 1:param.numIteration
    % 稀疏编码阶段
    if (param.errorFlag==0)     % 表示每个信号的稀疏系数个数是固定的
        
        CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, param.L);
    else                        % 表示信号的稀疏系数不固定，由误差项决定迭代是否停止
       
        CoefMatrix = OMPerr([FixedDictionaryElement,Dictionary],Data, param.errorGoal);
        param.L = 1;
    end
    % 字典更新阶段
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));   % 生成一个1到size(Dictionary,2)的随机的向量 
    for j = rPerm                           % 对字典的所有原子进行更新，更新顺序随机
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...    % 计算更新的稀疏系数
                                                                [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
                                                                CoefMatrix );
        Dictionary(:,j) = betterDictionaryElement;  % 更新字典原子
        if (param.preserveDCAtom)                   % 第一列原子为常数
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));  %归一化
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;                   % 计算替换向量的个数
    end

    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(size(FixedDictionaryElement,2)+1:end,:),Data); %检查原子的相关性
    
end
%========================================================================================================================================
output.CoefMatrix = CoefMatrix;                     %输出稀疏系数矩阵
Dictionary = [FixedDictionaryElement,Dictionary];   %输出训练字典

%========================================================= 以下为部分子函数 ==============================================================
%==========================================================================
%  函数功能: 对残差项进行奇异值分解，更新字典原子和稀疏系数
%  输入参数：Data - 训练信号
%           Dictionary - 字典
%           j - 稀疏系数的第j行
%           CoefMatrix - 系数矩阵
%  输出参数：betterDictionaryElement - 更新后的字典原子
%           CoefMatrix              - 系数矩阵
%           NewVectorAdded          - 新加入向量的个数
%==========================================================================
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix)

relevantDataIndices = find(CoefMatrix(j,:));    % 非零元在第j行的系数矩阵中的位置
if (length(relevantDataIndices)<1)              % 如果系数矩阵的第j行全为零
    ErrorMat = Data-Dictionary*CoefMatrix;      % 在已有的字典集下和系数下对data项的估计误差  
    ErrorNormVec = sum(ErrorMat.^2);            % 绝对误差
    [d,i] = max(ErrorNormVec);                  % d为矩阵每列的最大项，i为每个最大值所在的列 
    betterDictionaryElement = Data(:,i);        % 数据项的i列赋给betterDictionaryElement 
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);  % 归一化
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));                        % 将betterDictionaryElement中负的元素化为正的
    CoefMatrix(j,:) = 0;                        % 将系数矩阵第j行置0    
    NewVectorAdded = 1;                         %
    return;                                     % 注意，返回不再执行下面的语句
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices);%提取稀疏系数中非零项 将系数矩阵的第j行的非零项所在的列赋给tmpCoefMatrix  
tmpCoefMatrix(j,:) = 0;                           % the coeffitients of the element we now improve are not relevant.
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % 误差向量-对其进行奇异值分解
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);    %betterDictionaryElement 更新的字典原子
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';          % 更新稀疏系数


%==========================================================================
%  I_clearDictionary 
%  检查原子的相关性和稀疏系数的贡献
%==========================================================================
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;  %两个原子的相关性
T1 = 3;     %稀疏系数矩阵某一行系数大于某个阈值的个数
K=size(Dictionary,2);                           % 字典的列数
Er=sum((Data-Dictionary*CoefMatrix).^2,1);      % remove identical atoms 按列求和
G=Dictionary'*Dictionary; G = G-diag(diag(G));
for jj=1:1:K,
    if max(G(jj,:))>T2 || length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        [val,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=Dictionary'*Dictionary; G = G-diag(diag(G));
    end;
end;

%======================================================== end of the file ========================================================