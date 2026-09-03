function [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, ...
maxNumNeurons, decNumNeurons, p, critNum, delta, Xv, Yv, verbose)
% GMDHBUILD
% Builds a GMDH-type polynomial neural network using a simple
% layer-by-layer approach
%
% Call
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons, p, critNum, delta, Xv, Yv, verbose)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons, p, critNum, delta, Xv, Yv)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons, p, critNum, delta)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons, p, critNum)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons, p)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons,
%                   decNumNeurons)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore, maxNumNeurons)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs, inputsMore)
%   [model, time] = gmdhbuild(Xtr, Ytr, maxNumInputs)
%   [model, time] = gmdhbuild(Xtr, Ytr)
%
% Input
% Xtr, Ytr     : Training data points (Xtr(i,:), Ytr(i)), i = 1,...,n
% maxNumInputs : Maximum number of inputs for individual neurons - if set
%                to 3, both 2 and 3 inputs will be tried (default = 2)
% inputsMore   : Set to 0 for the neurons to take inputs only from the
%                preceding layer, set to 1 to take inputs also from the
%                original input variables (default = 1)
% maxNumNeurons: Maximal number of neurons in a layer (default = equal to
%                the number of the original input variables)
% decNumNeurons: In each following layer decrease the number of allowed
%                neurons by decNumNeurons until the number is equal to 1
%                (default = 0)
% p            : Degree of polynomials in neurons (allowed values are 2 and
%                3) (default = 2)
% critNum      : Criterion for evaluation of neurons and for stopping.
%                In each layer only the best neurons (according to the
%                criterion) are retained, and the rest are discarded.
%                (default = 2)
%                0 = use validation data (Xv, Yv)
%                1 = use validation data (Xv, Yv) as well as training data
%                2 = use Corrected Akaike's Information Criterion (AICC)
%                3 = use Minimum Description Length (MDL)
%                Note that both choices 0 and 1 correspond to the so called
%                "regularity criterion".
% delta        : How much lower the criterion value of the network's new
%                layer must be comparing the the network's preceding layer
%                (default = 0, which means that new layers will be added as
%                long as the value gets better (smaller))
% Xv, Yv       : Validation data points (Xv(i,:), Yv(i)), i = 1,...,nv
%                (used when critNum is equal to either 0 or 1)
% verbose      : Set to 0 for no verbose (default = 1)
%
% Output
% model        : GMDH model - a struct with the following elements:
%    numLayers     : Number of layers in the network
%    d             : Number of input variables in the training data set
%    maxNumInputs  : Maximal number of inputs for neurons
%    inputsMore    : See argument "inputsMore"
%    maxNumNeurons : Maximal number of neurons in a layer
%    p             : See argument "p"
%    critNum       : See argument "critNum"
%    layer         : Full information about each layer (number of neurons,
%                    indexes of inputs for neurons, matrix of exponents for
%                    polynomial, polynomial coefficients)
%                    Note that the indexes of inputs are in range [1..d] if
%                    an input is one of the original input variables, and
%                    in range [d+1..d+maxNumNeurons] if an input is taken
%                    from a neuron in the preceding layer.
% time         : Execution time (in seconds)
%
% Please give a reference to the software web page in any publication
% describing research performed using the software e.g., like this:
% Jekabsons G. GMDH-type Polynomial Neural Networks for Matlab, 2010,
% available at http://www.cs.rtu.lv/jekabsons/
 
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
 
display !!
if nargin < 2
    error('Too few input arguments.');
end
 
[n, d] = size(Xtr);
[ny, dy] = size(Ytr);
if (n < 2) || (d < 2) || (ny ~= n) || (dy ~= 1)
    error('Wrong training data sizes.');
end
 
if nargin < 3
    maxNumInputs = 2;
elseif (maxNumInputs ~= 2) && (maxNumInputs ~= 3)
    error('Number of inputs for neurons should be 2 or 3.');
end
if (d < maxNumInputs)
    error('Numbet of input variables in the data is lower than the number of inputs for individual neurons.');
end
if nargin < 4
    inputsMore = 1;
end
if (nargin < 5) || (maxNumNeurons <= 0)
    maxNumNeurons = d;
end
if maxNumNeurons > d * 2
    error('Too many neurons in a layer. Maximum is two times the number of input variables.');
end
if maxNumNeurons < 1
    error('Too few neurons in a layer. Minimum is 1.');
end
if (nargin < 6) || (decNumNeurons < 0)
    decNumNeurons = 0;
end
if nargin < 7
    p = 2;
elseif (p ~= 2) && (p ~= 3)
    error('Degree of individual neurons should be 2 or 3.');
end
if nargin < 8
    critNum = 2;
end
if any(critNum == [0,1,2,3]) == 0
    error('Only four values for critNum are available (0,1 - use validation data; 2 - AICC; 3 - MDL).');
end
if nargin < 9
    delta = 0;
end
if (nargin < 11) && (critNum <= 1)
    error('Evaluating the models in validation data requires validation data set.');
end
if (nargin >= 11) && (critNum <= 1)
    [nv, dv] = size(Xv);
    [nvy, dvy] = size(Yv);
    if (nv < 1) || (dv ~= d) || (nvy ~= nv) || (dvy ~= 1)
        error('Wrong validation data sizes.');
    end
end
if nargin < 12
    verbose = 1;
end
 
ws = warning('off');
if verbose ~= 0
    fprintf('Building GMDH-type neural network...\n');
end
tic;
 
if p == 2
    numTermsReal = 6 + 4 * (maxNumInputs == 3); %6 or 10 terms
else
    numTermsReal = 10 + 10 * (maxNumInputs == 3); %10 or 20 terms
end
 
Xtr(:, d+1:d+maxNumNeurons) = zeros(n, maxNumNeurons);
if critNum <= 1
    Xv(:, d+1:d+maxNumNeurons) = zeros(nv, maxNumNeurons);
end
 
%start the main loop and create layers
model.numLayers = 0;
while 1
 
    if verbose ~= 0
        fprintf('Building layer #%d...\n', model.numLayers + 1);
    end
 
    layer(model.numLayers + 1).numNeurons = 0;
    modelsTried = 0;
    layer(model.numLayers + 1).coefs = zeros(maxNumNeurons, numTermsReal);
 
    for numInputsTry = maxNumInputs:-1:2
 
        %create matrix of exponents for polynomials
        if p == 2
            numTerms = 6 + 4 * (numInputsTry == 3); %6 or 10 terms
            if numInputsTry == 2
                r = [0,0;0,1;1,0;1,1;0,2;2,0];
            else
                r = [0,0,0;0,0,1;0,1,0;1,0,0;0,1,1;1,0,1;1,1,0;0,0,2;0,2,0;2,0,0];
            end
        else
            numTerms = 10 + 10 * (numInputsTry == 3); %10 or 20 terms
            if numInputsTry == 2
                r = [0,0;0,1;1,0;1,1;0,2;2,0;1,2;2,1;0,3;3,0];
            else
                r = [0,0,0;0,0,1;0,1,0;1,0,0;0,1,1;1,0,1;1,1,0;0,0,2;0,2,0;2,0,0; ...
                     1,1,1;0,1,2;0,2,1;1,0,2;1,2,0;2,0,1;2,1,0;0,0,3;0,3,0;3,0,0];
            end
        end
 
        %create matrix of all combinations of inputs for neurons
        if model.numLayers == 0
            combs = nchoosek(1:1:d, numInputsTry);
        else
            if inputsMore == 1
                combs = nchoosek([1:1:d d+1:1:d+layer(model.numLayers).numNeurons], numInputsTry);
            else
                combs = nchoosek(d+1:1:d+layer(model.numLayers).numNeurons, numInputsTry);
            end
        end
        %delete all combinations in which none of the inputs are from the preceding layer
        if model.numLayers > 0
            i = 1;            
            while i <= size(combs,1)
                if all(combs(i,:) <= d)
                    combs(i,:) = [];
                else
                    i = i + 1;
                end
            end
        end
        
        makeEmpty = 1;
        
        %try all the combinations of inputs for neurons
        for i = 1 : size(combs,1)
 
            %create matrix for all polynomial terms
            Vals = ones(n, numTerms);
            if critNum <= 1
                Valsv = ones(nv, numTerms);
            end
            for idx = 2 : numTerms
                bf = r(idx, :);
                t = bf > 0;
                tmp = Xtr(:, combs(i,t)) .^ bf(ones(n, 1), t);
                if critNum <= 1
                    tmpv = Xv(:, combs(i,t)) .^ bf(ones(nv, 1), t);
                end
                if size(tmp, 2) == 1
                    Vals(:, idx) = tmp;
                    if critNum <= 1
                        Valsv(:, idx) = tmpv;
                    end
                else
                    Vals(:, idx) = prod(tmp, 2);
                    if critNum <= 1
                        Valsv(:, idx) = prod(tmpv, 2);
                    end
                end
            end
 
            %calculate coefficients and evaluate the network
            coefs = (Vals' * Vals) \ (Vals' * Ytr);
            modelsTried = modelsTried + 1;
            if ~isnan(coefs(1))
                predY = Vals * coefs;
                if critNum <= 1
                    predYv = Valsv * coefs;
                    if critNum == 0
                        crit = sqrt(mean((predYv - Yv).^2));
                    else
                        crit = sqrt(mean([(predYv - Yv).^2; (predY - Ytr).^2]));
                    end
                else
                    comp = complexity(layer, model.numLayers, maxNumNeurons, d, combs(i,:)) + size(coefs, 2);
                    if critNum == 2 %AICC
                        if (n-comp-1 > 0)
                            crit = n*log(mean((predY - Ytr).^2)) + 2*comp + 2*comp*(comp+1)/(n-comp-1);
                        else
                            coefs = NaN;
                        end
                    else %MDL
                        crit = n*log(mean((predY - Ytr).^2)) + comp*log(n);
                    end
                end
            end
 
            if ~isnan(coefs(1))
                %add the neuron to the layer if
                %1) the layer is not full;
                %2) the new neuron is better than an existing worst one.
                maxN = maxNumNeurons - model.numLayers * decNumNeurons;
                if maxN < 1, maxN = 1; end;
                if layer(model.numLayers + 1).numNeurons < maxN
                    %when the layer is not yet full
                    if (maxNumInputs == 3) && (numInputsTry == 2)
                        layer(model.numLayers + 1).coefs(layer(model.numLayers + 1).numNeurons+1, :) = [coefs' zeros(1,4+6*(p == 3))];
                        layer(model.numLayers + 1).inputs(layer(model.numLayers + 1).numNeurons+1, :) = [combs(i, :) 0];
                    else
                        layer(model.numLayers + 1).coefs(layer(model.numLayers + 1).numNeurons+1, :) = coefs;
                        layer(model.numLayers + 1).inputs(layer(model.numLayers + 1).numNeurons+1, :) = combs(i, :);
                    end
                    layer(model.numLayers + 1).comp(layer(model.numLayers + 1).numNeurons+1) = length(coefs);
                    layer(model.numLayers + 1).crit(layer(model.numLayers + 1).numNeurons+1) = crit;
                    layer(model.numLayers + 1).terms(layer(model.numLayers + 1).numNeurons+1).r = r;
                    if makeEmpty == 1
                        Xtr2 = [];
                        if critNum <= 1
                            Xv2 = [];
                        end
                        makeEmpty = 0;
                    end
                    Xtr2(:, layer(model.numLayers + 1).numNeurons+1) = predY;
                    if critNum <= 1
                        Xv2(:, layer(model.numLayers + 1).numNeurons+1) = predYv;
                    end
                    if (layer(model.numLayers + 1).numNeurons == 0) || ...
                       (layer(model.numLayers + 1).crit(worstOne) < crit)
                        worstOne = layer(model.numLayers + 1).numNeurons + 1;
                    end
                    layer(model.numLayers + 1).numNeurons = layer(model.numLayers + 1).numNeurons + 1;
                else
                    %when the layer is already full
                    if (layer(model.numLayers + 1).crit(worstOne) > crit)
                        if (maxNumInputs == 3) && (numInputsTry == 2)
                            layer(model.numLayers + 1).coefs(worstOne, :) = [coefs' zeros(1,4+6*(p == 3))];
                            layer(model.numLayers + 1).inputs(worstOne, :) = [combs(i, :) 0];
                        else
                            layer(model.numLayers + 1).coefs(worstOne, :) = coefs;
                            layer(model.numLayers + 1).inputs(worstOne, :) = combs(i, :);
                        end
                        layer(model.numLayers + 1).comp(worstOne) = length(coefs);
                        layer(model.numLayers + 1).crit(worstOne) = crit;
                        layer(model.numLayers + 1).terms(worstOne).r = r;
                        Xtr2(:, worstOne) = predY;
                        if critNum <= 1
                            Xv2(:, worstOne) = predYv;
                        end
                        [dummy, worstOne] = max(layer(model.numLayers + 1).crit);
                    end
                end
            end
 
        end
 
    end
 
    if verbose ~= 0
        fprintf('Neurons tried in this layer: %d\n', modelsTried);
        fprintf('Neurons included in this layer: %d\n', layer(model.numLayers + 1).numNeurons);
        if critNum <= 1
            fprintf('RMSE in the validation data of the best neuron: %f\n', min(layer(model.numLayers + 1).crit));
        else
            fprintf('Criterion value of the best neuron: %f\n', min(layer(model.numLayers + 1).crit));
        end
    end
 
    %stop the process if there are too few neurons in the new layer
    if ((inputsMore == 0) && (layer(model.numLayers + 1).numNeurons < 2)) || ...
       ((inputsMore == 1) && (layer(model.numLayers + 1).numNeurons < 1))
        if (layer(model.numLayers + 1).numNeurons > 0)
            model.numLayers = model.numLayers + 1;
        end
        break
    end
 
    %if the network got "better", continue the process
    if (layer(model.numLayers + 1).numNeurons > 0) && ...
       ((model.numLayers == 0) || ...
        (min(layer(model.numLayers).crit) - min(layer(model.numLayers + 1).crit) > delta) ) %(min(layer(model.numLayers + 1).crit) < min(layer(model.numLayers).crit)) )
        model.numLayers = model.numLayers + 1;
    else
        if model.numLayers == 0
            warning(ws);
            error('Failed.');
        end
        break
    end
 
    %copy the output values of this layer's neurons to the training
    %data matrix
    Xtr(:, d+1:d+layer(model.numLayers).numNeurons) = Xtr2;
    if critNum <= 1
        Xv(:, d+1:d+layer(model.numLayers).numNeurons) = Xv2;
    end
 
end
 
model.d = d;
model.maxNumInputs = maxNumInputs;
model.inputsMore = inputsMore;
model.maxNumNeurons = maxNumNeurons;
model.p = p;
model.critNum = critNum;
 
%only the neurons which are actually used (directly or indirectly) to
%compute the output value may stay in the network
[dummy best] = min(layer(model.numLayers).crit);
model.layer(model.numLayers).coefs(1,:) = layer(model.numLayers).coefs(best,:);
model.layer(model.numLayers).inputs(1,:) = layer(model.numLayers).inputs(best,:);
model.layer(model.numLayers).terms(1).r = layer(model.numLayers).terms(best).r;
model.layer(model.numLayers).numNeurons = 1;
if model.numLayers > 1
    for i = model.numLayers-1:-1:1 %loop through all the layers
        model.layer(i).numNeurons = 0;
        for k = 1 : layer(i).numNeurons %loop through all the neurons in this layer
            newNum = 0;
            for j = 1 : model.layer(i+1).numNeurons %loop through all the neurons which will stay in the next layer
                for jj = 1 : maxNumInputs %loop through all the inputs
                    if k == model.layer(i+1).inputs(j,jj) - d
                        if newNum == 0
                            model.layer(i).numNeurons = model.layer(i).numNeurons + 1;
                            model.layer(i).coefs(model.layer(i).numNeurons,:) = layer(i).coefs(k,:);
                            model.layer(i).inputs(model.layer(i).numNeurons,:) = layer(i).inputs(k,:);
                            model.layer(i).terms(model.layer(i).numNeurons).r = layer(i).terms(k).r;
                            newNum = model.layer(i).numNeurons + d;
                            model.layer(i+1).inputs(j,jj) = newNum;
                        else
                            model.layer(i+1).inputs(j,jj) = newNum;
                        end
                        break
                    end
                end
            end
        end
    end
end
 
time = toc;
warning(ws);
 
if verbose ~= 0
    fprintf('Done.\n');
    used = zeros(d,1);
    for i = 1 : model.numLayers
        for j = 1 : d
            if any(any(model.layer(i).inputs == j))
                used(j) = 1;
            end
        end
    end
    fprintf('Number of layers: %d\n', model.numLayers);
    fprintf('Number of used input variables: %d\n', sum(used));
    fprintf('Execution time: %0.2f seconds\n', time);
end
 
return
 
%====================  Auxiliary functions  ====================
 
function [comp] = complexity(layer, numLayers, maxNumNeurons, d, connections)
%calculates the complexity of the network given output neuron's connections
%(it is assumed that the complexity of a network is equal to the number of
%all polynomial terms in all it's neurons which are actually connected
%(directly or indirectly) to network's output)
comp = 0;
if numLayers == 0
    return
end
c = zeros(numLayers, maxNumNeurons);
for i = 1 : numLayers
    c(i, :) = layer(i).comp(:)';
end
%{
%unvectorized version:
for j = 1 : length(connections)
    if connections(j) > d
        comp = comp + c(numLayers, connections(j) - d);
        c(numLayers, connections(j) - d) = -1;
    end
end
%}
ind = connections > d;
if any(ind)
    comp = comp + sum(c(numLayers, connections(ind) - d));
    c(numLayers, connections(ind) - d) = -1;
end
%{
%unvectorized version:
for i = numLayers-1:-1:1
    for j = 1 : layer(i).numNeurons
        for k = 1 : layer(i+1).numNeurons
            if (c(i+1, k) == -1) && (c(i, j) > -1) && ...
               any(layer(i+1).inputs(k,:) == j + d)
                comp = comp + c(i, j);
                c(i, j) = -1;
            end
        end
    end
end
%}
for i = numLayers-1:-1:1
        for k = 1 : layer(i+1).numNeurons
            if c(i+1, k) == -1
                inp = layer(i+1).inputs(k,:);
                used = inp > d;
                if any(used)
                    ind = inp(used) - d;
                    ind = ind(c(i, ind) > -1);
                    if ~isempty(ind)
                        comp = comp + sum(c(i, ind));
                        c(i, ind) = -1;
                    end
                end
            end
        end
end
return














