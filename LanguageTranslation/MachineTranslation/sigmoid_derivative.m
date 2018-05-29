function y = sigmoid_derivative(hout)
    y = hout.*(1-hout)
end
