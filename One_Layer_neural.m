function [a] = One_Layer_neural(P, W, b, f, S)

    % 輸入: 
    % P: 輸入向量 (A vector of input) P = (P1, P2, …PR), R*1 輸入向量
    % W: 權重矩陣 (weight matrix)，Wjk 表示從輸入向量 p 的第 k 個輸入到第 j 個神經元的權重。S x R 權重矩陣
    % b: 閥值 (Threshold or called bias), S x 1 偏差向量
    % f(n): 轉化函數 (transfer function, also called activation function)
    % S: 神經元的數目 (Neurons)

    % Initialization
    n = zeros(size(b)); % Initializing n
    a = zeros(size(b)); % Initializing output a

    % Compute n for each neuron and then compute activation
    for i = 1:S
            % S 是神經元的數量，所以迴圈會從第一個神經元執行到第 S 個神經元。
        n(i) = W(i,:) * P + b(i);
            % 獲取權重矩陣 W 中第 i 行的所有元素，也就是第 i 個神經元的所有權重。
            % P 輸入向量

        a(i) = apply_transfer_function(n(i), f); % Note that we are passing n(i) not the whole n
    end
end

function val = apply_transfer_function(n, f)
    switch f
        case 'hard_limit'
            val = double(n >= 0);  % Returns 1 if n >= 0, else returns 0
        case 'symmetrical_hard_limit'
            val = double(n >= 0) * 2 - 1;  % Returns 1 if n >= 0, else returns -1
        case 'linear'
            val = n;  % Identity function
        case 'saturating_linear'
            val = min(max(n, 0), 1);  % Clipped between [0, 1]
        otherwise
            error('Unsupported Transfer function f(n).');
    end
end
