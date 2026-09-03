function Yq = gmdhpredict(model, Xq)
% GMDHPREDICT
% Predicts output values for the given query points Xq using a GMDH model
%
% Call
%   [Yq] = gmdhpredict(model, Xq)
%
% Input
% model     : GMDH model
% Xq        : Inputs of query data points (Xq(i,:)), i = 1,...,nq
%
% Output
% Yq        : Predicted outputs of query data points (Yq(i)), i = 1,...,nq
 
% This source code is tested with Matlab version 7.1 (R14SP3).
 
% =========================================================================
% GMDH-type polynomial neural network
% Version: 1.5
% Date: June 2, 2011
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2009-2011  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================
 
if nargin < 2
    error('Too few input arguments.');
end
if model.d ~= size(Xq, 2)
    error('The matrix should have the same number of columns as the matrix with which the network was built.');
end
 
[n, d] = size(Xq);
Yq = zeros(n, 1);
 
for q = 1 : n
    for i = 1 : model.numLayers
        if i ~= model.numLayers
            Xq_tmp = zeros(1, model.layer(i).numNeurons);
        end
        for j = 1 : model.layer(i).numNeurons
 
            %create matrix for all polynomial terms
            numTerms =  size(model.layer(i).terms(j).r,1);
            Vals = ones(numTerms,1);
            for idx = 2 : numTerms
                bf = model.layer(i).terms(j).r(idx, :);
                t = bf > 0;
                tmp = Xq(q, model.layer(i).inputs(j,t)) .^ bf(1, t);
                if size(tmp, 2) == 1
                    Vals(idx,1) = tmp;
                else
                    Vals(idx,1) = prod(tmp, 2);
                end
            end
 
            %predict output value
            predY = model.layer(i).coefs(j,1:numTerms) * Vals;
            if i ~= model.numLayers
                %Xq(q, d+j) = predY;
                Xq_tmp(j) = predY;
            else
                Yq(q) = predY;
            end
 
        end
        if i ~= model.numLayers
            Xq(q, d+1:d+model.layer(i).numNeurons) = Xq_tmp;
        end
    end
end
 
return
