__author__ = 'emulign'

import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('sources/train.csv', index_col=0)
predictions = pd.read_csv('pending-submissions/submission20150410-150349.csv', index_col=0)

# Histograms
#plt.hist(train.revenue, bins=20, label='Real')
plt.hist(predictions['Prediction'], bins=20, label='Predictions')

# Correlation between columns to revenue
#print train.P31.corr(train.revenue)
print 'Correation between column P1 with revenue: ',train.P1.corr(train.revenue)
print 'Correation between column P2 with revenue: ',train.P2.corr(train.revenue)
print 'Correation between column P3 with revenue: ',train.P3.corr(train.revenue)
print 'Correation between column P4 with revenue: ',train.P4.corr(train.revenue)
print 'Correation between column P5 with revenue: ',train.P5.corr(train.revenue)
print 'Correation between column P6 with revenue: ',train.P6.corr(train.revenue)
print 'Correation between column P7 with revenue: ',train.P7.corr(train.revenue)
print 'Correation between column P8 with revenue: ',train.P8.corr(train.revenue)
print 'Correation between column P9 with revenue: ',train.P9.corr(train.revenue)
print 'Correation between column P10 with revenue: ',train.P10.corr(train.revenue)
print 'Correation between column P11 with revenue: ',train.P11.corr(train.revenue)
print 'Correation between column P12 with revenue: ',train.P12.corr(train.revenue)
print 'Correation between column P13 with revenue: ',train.P13.corr(train.revenue)
print 'Correation between column P14 with revenue: ',train.P14.corr(train.revenue)
print 'Correation between column P15 with revenue: ',train.P15.corr(train.revenue)
print 'Correation between column P16 with revenue: ',train.P16.corr(train.revenue)
print 'Correation between column P17 with revenue: ',train.P17.corr(train.revenue)
print 'Correation between column P18 with revenue: ',train.P18.corr(train.revenue)
print 'Correation between column P19 with revenue: ',train.P19.corr(train.revenue)
print 'Correation between column P20 with revenue: ',train.P20.corr(train.revenue)
print 'Correation between column P21 with revenue: ',train.P21.corr(train.revenue)
print 'Correation between column P22 with revenue: ',train.P22.corr(train.revenue)
print 'Correation between column P23 with revenue: ',train.P23.corr(train.revenue)
print 'Correation between column P24 with revenue: ',train.P24.corr(train.revenue)
print 'Correation between column P25 with revenue: ',train.P25.corr(train.revenue)
print 'Correation between column P26 with revenue: ',train.P26.corr(train.revenue)
print 'Correation between column P27 with revenue: ',train.P27.corr(train.revenue)
print 'Correation between column P28 with revenue: ',train.P28.corr(train.revenue)
print 'Correation between column P29 with revenue: ',train.P29.corr(train.revenue)
print 'Correation between column P30 with revenue: ',train.P30.corr(train.revenue)
print 'Correation between column P31 with revenue: ',train.P31.corr(train.revenue)
print 'Correation between column P32 with revenue: ',train.P32.corr(train.revenue)
print 'Correation between column P33 with revenue: ',train.P33.corr(train.revenue)
print 'Correation between column P34 with revenue: ',train.P34.corr(train.revenue)
print 'Correation between column P35 with revenue: ',train.P35.corr(train.revenue)
print 'Correation between column P36 with revenue: ',train.P36.corr(train.revenue)
print 'Correation between column P37 with revenue: ',train.P37.corr(train.revenue)

# Correation between column P1 with revenue:  0.0742461337274
# Correation between column P2 with revenue:  0.192197246817
# Correation between column P3 with revenue:  -0.0153231464688
# Correation between column P4 with revenue:  0.0449175175545
# Correation between column P5 with revenue:  -0.0188057747696
# Correation between column P6 with revenue:  0.141060399365
# Correation between column P7 with revenue:  0.0626653500067
# Correation between column P8 with revenue:  -0.0755908462279
# Correation between column P9 with revenue:  -0.0362232357842
# Correation between column P10 with revenue:  -0.0593657440609
# Correation between column P11 with revenue:  0.0985916539378
# Correation between column P12 with revenue:  -0.0484373550254
# Correation between column P13 with revenue:  -0.0918991828865
# Correation between column P14 with revenue:  0.016069707054
# Correation between column P15 with revenue:  0.00698251287683
# Correation between column P16 with revenue:  -0.0265334930029
# Correation between column P17 with revenue:  0.0727912675602
# Correation between column P18 with revenue:  -0.0171454601192
# Correation between column P19 with revenue:  0.0434572952902
# Correation between column P20 with revenue:  0.0256816137238
# Correation between column P21 with revenue:  0.107987369477
# Correation between column P22 with revenue:  0.0780017977234
# Correation between column P23 with revenue:  0.0538465571
# Correation between column P24 with revenue:  0.0158066614565
# Correation between column P25 with revenue:  0.0385443912402
# Correation between column P26 with revenue:  -0.00540674020833
# Correation between column P27 with revenue:  -0.00995291880275
# Correation between column P28 with revenue:  0.154013044159
# Correation between column P29 with revenue:  -0.100174767696
# Correation between column P30 with revenue:  -0.0432835542938
# Correation between column P31 with revenue:  -0.0240876896822
# Correation between column P32 with revenue:  -0.0577873667504
# Correation between column P33 with revenue:  -0.016092225403
# Correation between column P34 with revenue:  -0.0552714931596
# Correation between column P35 with revenue:  -0.033626347057
# Correation between column P36 with revenue:  -0.0371737840635
# Correation between column P37 with revenue:  -0.00346807980632


# Plots
# plt.figure(1)
# plt.plot(train.revenue, train.P4, 'ro')
# plt.xlabel('Revenue')
# plt.ylabel('P1')
# plt.legend()
#
# plt.figure(2)
# plt.plot(train.revenue, train.P5, 'ro')
# plt.xlabel('Revenue')
# plt.ylabel('P2')
# plt.legend()
#
#
# plt.figure(3)
# plt.plot(train.revenue, train.P6, 'ro')
# plt.xlabel('Revenue')
# plt.ylabel('P3')
# plt.legend()
plt.show()