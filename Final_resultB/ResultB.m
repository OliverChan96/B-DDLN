%% Gathering labels
clc;clear all;close all;
Label=[];
load('resultBlinear.mat', 'label_test');
Label(:,1)=label_test;%Predicting labels of CDNN-Linear
load('resultBtanh.mat', 'label_test');
Label(:,2)=label_test;%Predicting labels of CDNN-Tanh
load('resultBsinh.mat', 'label_test');
Label(:,3)=label_test;%Predicting labels of CDNN-Sinh
clear label_test;

%Voting method
for i=1:size(Label,1)
    a=Label(i,1:3);
    b=0:max(a);
    c=histc(a,b);
    [max_num, max_index]=max(c);
    Label(i,4)=b(max_index);%Voting results (predicting labels)
end

Label(:,5)=csvread('verLabels.csv');%Actual labels

%% Performances
%1.Confusion Matrix
% TP=Confusion_Matrix(1,1);FN=Confusion_Matrix(1,2);
% FP=Confusion_Matrix(2,1);TN=Confusion_Matrix(2,2);
cf_linear=confusionmat(Label(:,5),Label(:,1));
cf_tanh=confusionmat(Label(:,5),Label(:,2));
cf_sinh=confusionmat(Label(:,5),Label(:,3));
cf_bagging=confusionmat(Label(:,5),Label(:,4));

%2.Accuracy Rate
%Accuracyrate=(TP+TN)/(TP+TN+FP+FN);
Accuracyrate_linear=(cf_linear(1,1)+cf_linear(2,2))/sum(sum(cf_linear));
Accuracyrate_tanh=(cf_tanh(1,1)+cf_tanh(2,2))/sum(sum(cf_tanh));
Accuracyrate_sinh=(cf_sinh(1,1)+cf_sinh(2,2))/sum(sum(cf_sinh));
Accuracyrate_bagging=(cf_bagging(1,1)+cf_bagging(2,2))/sum(sum(cf_bagging));

%3.TP Rate
%TPR=TP/(TP+FN);
%TPR=Recall=Sensitivity
TPR_linear=cf_linear(1,1)/(cf_linear(1,1)+cf_linear(1,2));
TPR_tanh=cf_tanh(1,1)/(cf_tanh(1,1)+cf_tanh(1,2));
TPR_sinh=cf_sinh(1,1)/(cf_sinh(1,1)+cf_sinh(1,2));
TPR_bagging=cf_bagging(1,1)/(cf_bagging(1,1)+cf_bagging(1,2));

%4.FP Rate
%FPR=FP/(FP+TN);
FPR_linear=cf_linear(2,1)/(cf_linear(2,1)+cf_linear(2,2));
FPR_tanh=cf_tanh(2,1)/(cf_tanh(2,1)+cf_tanh(2,2));
FPR_sinh=cf_sinh(2,1)/(cf_sinh(2,1)+cf_sinh(2,2));
FPR_bagging=cf_bagging(2,1)/(cf_bagging(2,1)+cf_bagging(2,2));

%5.Precision
%Precision=TP/(TP+FP);
Precision_linear=cf_linear(1,1)/(cf_linear(1,1)+cf_linear(2,1));
Precision_tanh=cf_tanh(1,1)/(cf_tanh(1,1)+cf_tanh(2,1));
Precision_sinh=cf_sinh(1,1)/(cf_sinh(1,1)+cf_sinh(2,1));
Precision_bagging=cf_bagging(1,1)/(cf_bagging(1,1)+cf_bagging(2,1));

%6.Specificity
%Specificity=TN/(TN+FP)
Specificity_linear=cf_linear(2,2)/(cf_linear(2,2)+cf_linear(2,1));
Specificity_tanh=cf_tanh(2,2)/(cf_tanh(2,2)+cf_tanh(2,1));
Specificity_sinh=cf_sinh(2,2)/(cf_sinh(2,2)+cf_sinh(2,1));
Specificity_bagging=cf_bagging(2,2)/(cf_bagging(2,2)+cf_bagging(2,1));

%7.F-Measure(F1 Score)
%F1 Score=2*Precision*TPR/(Precision+TPR)
F1_Score_linear=2*Precision_linear*TPR_linear/(Precision_linear+TPR_linear);
F1_Score_tanh=2*Precision_tanh*TPR_tanh/(Precision_tanh+TPR_tanh);
F1_Score_sinh=2*Precision_sinh*TPR_sinh/(Precision_sinh+TPR_sinh);
F1_Score_bagging=2*Precision_bagging*TPR_bagging/(Precision_bagging+TPR_bagging);

%8.Kappa Value
%Kappa Value=(po-pe)/(1-pe)
%po=Accuracyrate
%pe=(sum(Confusion_Matrix(1,:))*sum(Confusion_Matrix(:,1))+sum(Confusion_Matrix(2,:))*sum(Confusion_Matrix(:,2)))/(sum(sum(Confusion_Matrix)).^2)
po_linear=Accuracyrate_linear;
pe_linear=(sum(cf_linear(1,:))*sum(cf_linear(:,1))+sum(cf_linear(2,:))*sum(cf_linear(:,2)))/(sum(sum(cf_linear)).^2);
Kappa_Value_linear=(po_linear-pe_linear)/(1-pe_linear);

po_tanh=Accuracyrate_tanh;
pe_tanh=(sum(cf_tanh(1,:))*sum(cf_tanh(:,1))+sum(cf_tanh(2,:))*sum(cf_tanh(:,2)))/(sum(sum(cf_tanh)).^2);
Kappa_Value_tanh=(po_tanh-pe_tanh)/(1-pe_tanh);

po_sinh=Accuracyrate_sinh;
pe_sinh=(sum(cf_sinh(1,:))*sum(cf_sinh(:,1))+sum(cf_sinh(2,:))*sum(cf_sinh(:,2)))/(sum(sum(cf_sinh)).^2);
Kappa_Value_sinh=(po_sinh-pe_sinh)/(1-pe_sinh);

po_bagging=Accuracyrate_bagging;
pe_bagging=(sum(cf_bagging(1,:))*sum(cf_bagging(:,1))+sum(cf_bagging(2,:))*sum(cf_bagging(:,2)))/(sum(sum(cf_bagging)).^2);
Kappa_Value_bagging=(po_bagging-pe_bagging)/(1-pe_bagging);

figure(1);
plotconfusion((Label(:,1))',(Label(:,5))');
title('Confusion Matrix: CDNN-Linear');
xlabel('Predicted Class','FontWeight','bold');
ylabel('Actual Class','FontWeight','bold');

figure(2);
plotconfusion((Label(:,2))',(Label(:,5))');
title('Confusion Matrix: CDNN-Tanh');
xlabel('Predicted Class','FontWeight','bold');
ylabel('Actual Class','FontWeight','bold');

figure(3);
plotconfusion((Label(:,3))',(Label(:,5))');
title('Confusion Matrix: CDNN-sinh');
xlabel('Predicted Class','FontWeight','bold');
ylabel('Actual Class','FontWeight','bold');

figure(4);
plotconfusion((Label(:,4))',(Label(:,5))');
title('Confusion Matrix: Bagging CDNN');
xlabel('Predicted Class','FontWeight','bold');
ylabel('Actual Class','FontWeight','bold');

%9.ROC and AUC
[fp_rate,tp_rate,T,auc_linear]=perfcurve(Label(:,5),Label(:,1),1);
figure(5);
h=plot(fp_rate,tp_rate,'r-','Linewidth',2);
hold on;
plot(0:0.01:1,0:0.01:1,'k--','Linewidth',1);
set(gca,'XTick',[0:0.1:1]);
set(gca,'YTick',[0:0.1:1]);
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
grid on;
legend([h],['Linear type of CDLM',sprintf('\n'),sprintf('AUC = %.4f',auc_linear)],'Location','SouthEast');

[fp_rate,tp_rate,T,auc_tanh]=perfcurve(Label(:,5),Label(:,2),1);
figure(6);
h=plot(fp_rate,tp_rate,'g-','Linewidth',2,'Color',[0,0.5,0]);
hold on;
plot(0:0.01:1,0:0.01:1,'k--','Linewidth',1);
set(gca,'XTick',[0:0.1:1]);
set(gca,'YTick',[0:0.1:1]);
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
grid on;
legend([h],['Tanh type of CDLM',sprintf('\n'),sprintf('AUC = %.4f',auc_tanh)],'Location','SouthEast');

[fp_rate,tp_rate,T,auc_sinh]=perfcurve(Label(:,5),Label(:,3),1);
figure(7);
h=plot(fp_rate,tp_rate,'b-','Linewidth',2);
hold on;
plot(0:0.01:1,0:0.01:1,'k--','Linewidth',1);
set(gca,'XTick',[0:0.1:1]);
set(gca,'YTick',[0:0.1:1]);
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
grid on;
legend([h],['Sinh type of CDLM',sprintf('\n'),sprintf('AUC = %.4f',auc_sinh)],'Location','SouthEast');

[fp_rate,tp_rate,T,auc_bagging]=perfcurve(Label(:,5),Label(:,4),1);
figure(8);
h=plot(fp_rate,tp_rate,'Linewidth',2,'Color',[0.5,0,0.5]);
hold on;
plot(0:0.01:1,0:0.01:1,'k--','Linewidth',1);
set(gca,'XTick',[0:0.1:1]);
set(gca,'YTick',[0:0.1:1]);
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
grid on;
legend([h],['B-CDLM',sprintf('\n'),sprintf('AUC = %.4f',auc_bagging)],'Location','SouthEast');
