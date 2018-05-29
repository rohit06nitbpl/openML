function y = softmaxseterror(a,index)
  a = -a;
  a(index) = 1+a(index);
  y = a;
end