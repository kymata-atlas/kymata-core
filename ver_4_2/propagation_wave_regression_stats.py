import numpy as np
import scipy.stats.distributions as ssd

def ols_regress(X,y):
    # ordinary least squares regression
    p = np.linalg.pinv(X.transpose().dot(X))
    hat = X.dot(p).dot(X.transpose())
    beta_hat = p.dot(X.transpose()).dot(y)
    fitted = hat.dot(y)
    residuals = y - fitted
    total_ssq = np.sum((y - np.mean(y))**2)
    residual_ssq = np.sum(residuals**2)
    r = np.sqrt(1-residual_ssq/total_ssq)
    dfe = X.shape[0]-np.size(beta_hat)
    sigma_squared_hat = residual_ssq/dfe
    return beta_hat, r, dfe, sigma_squared_hat, p, hat

def go_for_it_simple(data):
    n = data.shape[0]
    lag = data[:,0].reshape((n,1))
    position = data[:,1]
    X = np.concatenate((np.ones((n,1)), lag), axis=1)
    beta_hat, r, dfe, sigma_squared_hat, p, hat = ols_regress(X,position)
    c=np.diag(p)
    intercept, slope = beta_hat
    se_intercept, se_slope = np.sqrt(c*sigma_squared_hat)
    t_slope = slope/se_slope
    p_slope = 2*(1-ssd.t.cdf(t_slope,dfe))
    print 'Slope', slope, '+/-', se_slope, 't', t_slope, 'P', p_slope
    print 'Intercept', intercept
    # , '+/-', se_intercept, 't', intercept/se_intercept
    print 'd.f.', dfe
    print 'R-squared', r**2
    return

def go_for_it_two_slopes_two_intercepts(data1, data2):
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    lag = np.concatenate((data1[:,0], data2[:,0])).reshape((n1+n2,1))
    position = np.concatenate((data1[:,1],data2[:,1]))
    group = np.concatenate((np.zeros(n1),np.ones(n2))).reshape((n1+n2,1))
    X = np.concatenate((np.ones((n1+n2,1)),
                        group,
                        lag,
                        lag*group), axis=1)
    beta_hat, r, dfe, sigma_squared_hat, p, hat = ols_regress(X, position)
    c = np.diag(p)
    delta_slope = beta_hat[3]
    se_param = np.sqrt(c[3]*sigma_squared_hat)
    t_param = delta_slope/se_param
    p_param = 2*(1-ssd.t.cdf(t_param,dfe))
    print 'Difference of slopes', delta_slope, '+/-', se_param, 't', t_param, 'P', p_param
    print 'd.f.', dfe
    print 'R-squared', r**2
    return

left_data = np.array(
[[326,	21.6499096586562],
[311,	-22.0172696503240],
[301,	-40.2008035031857],
[326,	19.3909063969317],
[331,	-21.6094082759697],
[276,	-38.7147445115902],
[326,	-35.3115072263743],
[326,	-22.0305924854103],
[311,	-24.3453633183723],
[286,	-31.8182127980352],
[341,	-11.2989386023755],
[326,	20.5172612044719],
[331,	-22.9590185055504],
[326,	-21.3797210195034],
[301,	-41.1826849536728],
[311,	-33.7628800662500],
[281,	-28.1858437836269],
[321,	-11.1871331620695],
[336,	-15.7285049061699],
[321,	-16.0555943405241],
[321,	-37.1622186417273],
[301,	-35.6798772157122],
[326,	-33.9091984838121],
[326,	-33.8109188657182],
[311,	-10.1507411028485],
[321,	-22.0094580224390],
[311,	-20.1785868414805],
[321,	-20.5093946211882],
[321,	-18.5524859600811],
[326,	-24.6523853276806],
[326,	-23.6458455747303],
[261,	-31.6981503433740],
[321,	-28.0990219046294],
[326,	-26.4730722357751],
[321,	-11.2130671393985],
[331,	-18.2615344010846],
[311,	-15.0007375914349]],dtype='float')


 #   [[331,	-21.51],
  #   [321,	-12.9898],
#     [296,	-41.2161],
#     [301,	-35.93],
#     [331,	-17.96],
#     [326,	-26.39],
#     [276,	-34.92],
#     [326,	-31.87],
#     [281,	-28.3199],
#     [326,	-23.445],
#     [331,	-18.06],
#     [341,	-14.0034]],dtype='float')

right_data = np.array(
[[276,	-3.98261513714882],
[276,	-22.5435696076551],
[281,	-17.5986650196869],
[316,	8.29655684263744],
[271,	-26.4422649758606],
[261,	-25.0807094210261],
[261,	-43.0354753584847],
[286,	3.27167270691874],
[291,	-18.9527152324492],
[261,	-35.5742757697964],
[306,	-4.09303264029751],
[266,	-19.2266003305786],
[261,	-15.7974396350720],
[276,	-19.8101951929228],
[266,	14.2109908548913],
[306,	-12.4673731701721],
[316,	-9.55206829450029],
[256,	-27.5248066052511],
[271,	-21.5343496828266],
[286,	-26.0231678576214],
[286,	-23.7822788752857],
[281,	-24.6350566755164],
[281,	-22.9399948605938],
[261,	-43.5557448116250],
[276,	11.3130673353911],
[281,	-16.8108415918417],
[276,	-14.3816100566747],
[306,	-15.3027166980551],
[301,	-12.1253879578736],
[316,	-11.8150555601760],
[316,	-10.1587744480777],
[316,	-7.73126187241135],
[266,	-10.3048712815580],
[261,	-25.7435755894272],
[261,	-26.8440031990028],
[276,	5.36575246920449],
[276,	-18.4819992895692],
[286,	-16.1794153929036],
[291,	-16.7037387147806],
[261,	-33.3586905877560],
[261,	-17.1664836290235],
[266,	-17.6317407052067],
[271,	-18.5243345914218],
[261,	-22.5820055533723],
[266,	-20.7219585406287],
[276,	-19.6095910402577],
[266,	-21.5354303263927],
[271,	-30.9173440997722],
[266,	15.8121594664816],
[266,	-12.9600092341311],
[266,	-11.3988526489036],
[276,	-9.97226534146280],
[276,	-13.5834585751486],
[301,	-10.9519428792228],
[306,	-8.64953785968384],
[306,	-6.90658701458804]],dtype='float')


#    [[276,	-17.86],
#     [281,	-25.57],
#     [286,	-8.3],
#     [316,	-9.63],
#     [261,	-41.42],
#     [261,	-32.28],
#     [271,	-24.56],
#     [261,	-28.62],
#     [266,	-16.64],
#     [261,	-20.50],
#     [276,	-21.51],
#     [286,	-18.47],
#     [276,	-4.36],
#     [286,	-13.49],
#     [266,	-12.889],
#     [306,	-8.92],
#     [316,	-5.06],
#     [306,	-1.107]],dtype='float')

# simple regression on left hemi data
print '\n>>> Simple regression of left hemi data'
go_for_it_simple(left_data)

# simple regression on right hemi data
print '\n>>> Simple regression of right hemi data'
go_for_it_simple(right_data)

# fit model with two intercepts and two slopes to the combined left and
# right data
print '\n>>> regression with two slopes and two intercepts'
go_for_it_two_slopes_two_intercepts(left_data,right_data)

