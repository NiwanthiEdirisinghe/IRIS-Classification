clear all;
clc;

%load training data
input = xlsread('Train.xlsx');

%Normalization

for a = 1:4
    colmax = max(input(1:75,a:a));
    colmin = min(input(1:75,a:a));
    for b = 1:75
        input(b,a) = (input(b,a)-colmin)/(colmax-colmin);
    end
end

%Backpropagation algorithm
 
%initialize all vectors in th network
w1 = [4 -8 -6 3; 0 -61 15 -8; -38 -15 98 0];
w2 = [-25 -12 5 ];

noEphocs = 500;        %no of ephocs
meanSqureError = zeros([noEphocs:1]);
for ephoc = 1:noEphocs
    SqureError = 0;
    for notrainData = 1:75
        p = input( notrainData:notrainData , 1:4);
        p = p';
        n1 = w1*p;      %net input of the 1st layer
        a1 = logsig(n1);        %output of the Second layer
        n2 = w2*a1;     %net input of the second layer
        a2 = logsig(n2);        %output of the second layer
        e = input(notrainData,5) - a2;      %error of the final layer
        SqureError = SqureError + e^2;
        
        %consider tollerence
        if(abs(e)<=0.01) 
            e = 0;
        end
        
        %Back propagate the errors
        
        %calculations of error of each node
        err2 = a2*(1-a2)*e;     %error of  the node in the output layer
        f = [a1(1,1)*(1-a1(1,1)) 0 0;
             0 a1(2,1)*(1-a1(2,1)) 0;
             0 0 a1(3,1)*(1-a1(3,1))];
        err1 = f*err2*w2';      %errors of nodes in the hidden layer
        
        %calculations for weight updating
        learningRate = 0.7;
        deltaw2 = learningRate*err2*a1';    %weight increment of output layer
        deltaw1 = learningRate*err1*p';     %weight increment of hidden layer
        
        %updating weights
        w1 = w1 + deltaw1;
        w2 = w2 + deltaw2;
       
    end
    meanSqureError(ephoc,1) = SqureError/notrainData;    %Calculte mean squre error
end

%testing
testData = xlsread('Test.xlsx');

for a = 1:4
    colmax = max(testData(1:75,a:a));
    colmin = min(testData(1:75,a:a));
    for b = 1:75
       testData(b,a) = (testData(b,a)-colmin)/(colmax-colmin);
    end
end
testOutput = zeros([75,1]);
for a = 1:75
    p = testData(a, 1:4);
        p = p';
        n1 = w1*p;  
        a1 = logsig(n1);    
        n2 = w2*a1; 
        a2 = logsig(n2);
        testOutput(a,1)= a2;
end

%plot Graphs
figure(1);
hold on;
scatter(testOutput(1:25,1),1:25,'filled');
scatter(testOutput(26:50,1),26:50,'filled');
scatter(testOutput(51:75,1),51:75,'filled');
xlabel('Test output value');
ylabel('i th test data')
hold off;
figure(2);
line(1:noEphocs , meanSqureError);
xlabel('No of Ephocs');
ylabel('Mean Squre Error');
