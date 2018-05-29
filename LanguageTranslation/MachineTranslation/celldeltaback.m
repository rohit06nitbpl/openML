function y = celldeltaback(del,hout,whh)
  g = hout.*(1-hout).*del;
  y = whh'*g;
end
