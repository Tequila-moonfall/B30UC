%==========================================================================
%  ��������: ��K-SVD�㷨ѵ���ֵ�
%  ���������Data  - ����N��nά�źŵĴ�СΪnxN�ľ���
%           param - ��������
%                 - K����Ҫѵ�����ֵ�Ԫ�ظ���
%                 - numIteration�������Ĵ���
%                 - errorFlag������0���ʾÿ���źŵ�ϡ��ϵ�������̶�����Ҫ���ò���L��
%                              �����ʾ�źŵ�ϡ��ϵ�����̶�����Ҫ���ò���errorGoal��
%                 - preserveDCAtom������1���ֵ�ĵ�һ��ԭ��Ϊ������
%                 - InitializationMethod����DataElements�����źű�����Ϊ�ֵ�
%                                         ��GivenMatrix���ø����ľ�����Ϊ��ʼ�ֵ�
%                 - initialDictionary����ʼ���ֵ�
%  ���������Dictionary - ѵ���õ����ֵ䣬��СΪnxparam.K
%           output     - ����ṹ��
%                      - CoefMatrix:ϡ�����
%                      - numCoef��ƽ����ʾÿ��ԭ�ӵ�ϵ������
%==========================================================================

function [Dictionary,output] = KSVD(    ...
                                        Data,... % ����N��nά�źŵĴ�СΪnxN�ľ���
                                        param)   % �������

if (isfield(param,'errorFlag')==0)      % ��ʾÿ���źŵ�ϡ��ϵ�������̶�
    param.errorFlag = 0;
end

if (param.preserveDCAtom>0)             % ����˵������Ҫ��˱�������1��ִ�����²�����ʵ���������������0��ִ�в�����
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));% ����һ��Ϊ������Ϊ�˸��õı�ʾ��Ȼͼ��
else
    FixedDictionaryElement = [];% ���ٿռ�
end

if (size(Data,2) < param.K)     % ������K�����źŵĸ����������ݼ���Ϊ�ֵ伯 
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements')) 
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);% �����ݼ���1��param.K-param.preserveDCAtom������Ϊ�ֵ伯  
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))   % 
    Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,1:param.K-param.preserveDCAtom);% ��initialDictionary��1��param.K-param.preserveDCAtom����Ϊ�ֵ伯
end

if (param.preserveDCAtom)   % �ֵ��һ��ԭ��Ϊ����
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end
% ��һ���ֵ�
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary))); %��һ��
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.% ���ֵ伯�е�ÿ��Ԫ�صĻ�Ϊ����
totalErr = zeros(1,param.numIteration);

%================================= ��ʼK-SVD�㷨 ===========================
for iterNum = 1:param.numIteration
    % ϡ�����׶�
    if (param.errorFlag==0)     % ��ʾÿ���źŵ�ϡ��ϵ�������ǹ̶���
        
        CoefMatrix = OMP([FixedDictionaryElement,Dictionary],Data, param.L);
    else                        % ��ʾ�źŵ�ϡ��ϵ�����̶������������������Ƿ�ֹͣ
       
        CoefMatrix = OMPerr([FixedDictionaryElement,Dictionary],Data, param.errorGoal);
        param.L = 1;
    end
    % �ֵ���½׶�
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));   % ����һ��1��size(Dictionary,2)����������� 
    for j = rPerm                           % ���ֵ������ԭ�ӽ��и��£�����˳�����
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...    % ������µ�ϡ��ϵ��
                                                                [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
                                                                CoefMatrix );
        Dictionary(:,j) = betterDictionaryElement;  % �����ֵ�ԭ��
        if (param.preserveDCAtom)                   % ��һ��ԭ��Ϊ����
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));  %��һ��
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;                   % �����滻�����ĸ���
    end

    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(size(FixedDictionaryElement,2)+1:end,:),Data); %���ԭ�ӵ������
    
end
%========================================================================================================================================
output.CoefMatrix = CoefMatrix;                     %���ϡ��ϵ������
Dictionary = [FixedDictionaryElement,Dictionary];   %���ѵ���ֵ�

%========================================================= ����Ϊ�����Ӻ��� ==============================================================
%==========================================================================
%  ��������: �Բв����������ֵ�ֽ⣬�����ֵ�ԭ�Ӻ�ϡ��ϵ��
%  ���������Data - ѵ���ź�
%           Dictionary - �ֵ�
%           j - ϡ��ϵ���ĵ�j��
%           CoefMatrix - ϵ������
%  ���������betterDictionaryElement - ���º���ֵ�ԭ��
%           CoefMatrix              - ϵ������
%           NewVectorAdded          - �¼��������ĸ���
%==========================================================================
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix)

relevantDataIndices = find(CoefMatrix(j,:));    % ����Ԫ�ڵ�j�е�ϵ�������е�λ��
if (length(relevantDataIndices)<1)              % ���ϵ������ĵ�j��ȫΪ��
    ErrorMat = Data-Dictionary*CoefMatrix;      % �����е��ֵ伯�º�ϵ���¶�data��Ĺ������  
    ErrorNormVec = sum(ErrorMat.^2);            % �������
    [d,i] = max(ErrorNormVec);                  % dΪ����ÿ�е�����iΪÿ�����ֵ���ڵ��� 
    betterDictionaryElement = Data(:,i);        % �������i�и���betterDictionaryElement 
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);  % ��һ��
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));                        % ��betterDictionaryElement�и���Ԫ�ػ�Ϊ����
    CoefMatrix(j,:) = 0;                        % ��ϵ�������j����0    
    NewVectorAdded = 1;                         %
    return;                                     % ע�⣬���ز���ִ����������
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices);%��ȡϡ��ϵ���з����� ��ϵ������ĵ�j�еķ��������ڵ��и���tmpCoefMatrix  
tmpCoefMatrix(j,:) = 0;                           % the coeffitients of the element we now improve are not relevant.
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % �������-�����������ֵ�ֽ�
[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);    %betterDictionaryElement ���µ��ֵ�ԭ��
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';          % ����ϡ��ϵ��


%==========================================================================
%  I_clearDictionary 
%  ���ԭ�ӵ�����Ժ�ϡ��ϵ���Ĺ���
%==========================================================================
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;  %����ԭ�ӵ������
T1 = 3;     %ϡ��ϵ������ĳһ��ϵ������ĳ����ֵ�ĸ���
K=size(Dictionary,2);                           % �ֵ������
Er=sum((Data-Dictionary*CoefMatrix).^2,1);      % remove identical atoms �������
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