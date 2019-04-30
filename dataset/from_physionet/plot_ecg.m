%segment = ecg100(1:3000);

ecg100 = VarName3;
DetectionTime100 = VarName4;

plot(ecg100)

x = 1:size(ecg100,1);
y = zeros(size(x));

y(DetectionTime100) = ecg100(DetectionTime100);

hold on

plot(y)