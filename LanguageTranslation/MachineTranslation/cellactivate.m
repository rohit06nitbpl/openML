function y = cellactivate(whh,whx,h,x,b)
  y = sigmoid(whh*h+whx*x+b);
end