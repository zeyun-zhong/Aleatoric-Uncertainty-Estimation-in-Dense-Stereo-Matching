function [optAUC] = getOptAUC(Epsilon)

optAUC = Epsilon + (1-Epsilon) * log(1-Epsilon);

end

