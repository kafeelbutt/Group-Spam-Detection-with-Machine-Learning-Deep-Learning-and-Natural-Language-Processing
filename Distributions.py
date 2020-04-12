import Features
from fitter import Fitter
import scipy.stats as sc   

def calculate_rating_deviation_distribution(reviewers_df,cdfs):
    sample=reviewers_df['rating_deviation'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_rating=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_rating

def calculate_average_helpfulness_distribution(reviewers_df,cdfs):
    sample=reviewers_df['avg_helpfulness'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_avg=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_avg

def calculate_burst_ratio_distribution(reviewers_df,cdfs):
    sample=reviewers_df['burst_ratio'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_ratio=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_ratio

def calculate_number_reviews_distribution(reviewers_df,cdfs):
    sample=reviewers_df['num_of_reviews'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_reviews=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_reviews

def calculate_text_similarity_distribution(reviewers_df,cdfs):
    sample=reviewers_df['similarity_index'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_index=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_index

def calculate_product_count_distribution(reviewers_df,cdfs):
    sample=reviewers_df['common_products'].tolist()
    f = Fitter(sample, distributions=cdfs)
    f.fit()
    f.summary()
    best=f.get_best()
    key=list(best.keys())[0]
    dist=eval("sc."+key)
    distribution_common=dist.pdf(sample,*(f.fitted_param[key]))
    return distribution_common
    

if __name__=='__main__':
    
    cdfs = [
    "norm",            #Normal (Gaussian)
    "alpha",           #Alpha
    "anglit",          #Anglit
    "arcsine",         #Arcsine
    "beta",            #Beta
    "betaprime",       #Beta Prime
    "bradford",        #Bradford
    "burr",            #Burr
    "cauchy",          #Cauchy
    "chi",             #Chi
    "chi2",            #Chi-squared
    "cosine",          #Cosine
    "dgamma",          #Double Gamma
    "dweibull",        #Double Weibull
    "erlang",          #Erlang
    "expon",           #Exponential
    "exponweib",       #Exponentiated Weibull
    "exponpow",        #Exponential Power
    "fatiguelife",     #Fatigue Life (Birnbaum-Sanders)
    "foldcauchy",      #Folded Cauchy
    "f",               #F (Snecdor F)
    "fisk",            #Fisk
    "foldnorm",        #Folded Normal
    "frechet_r",       #Frechet Right Sided, Extreme Value Type II
    "frechet_l",       #Frechet Left Sided, Weibull_max
    "gamma",           #Gamma
    "gausshyper",      #Gauss Hypergeometric
    "genexpon",        #Generalized Exponential
    "genextreme",      #Generalized Extreme Value
    "gengamma",        #Generalized gamma
    "genlogistic",     #Generalized Logistic
    "genpareto",       #Generalized Pareto
    "genhalflogistic", #Generalized Half Logistic
    "gilbrat",         #Gilbrat
    "gompertz",        #Gompertz (Truncated Gumbel)
    "gumbel_l",        #Left Sided Gumbel, etc.
    "gumbel_r",        #Right Sided Gumbel
    "halfcauchy",      #Half Cauchy
    "halflogistic",    #Half Logistic
    "halfnorm",        #Half Normal
    "hypsecant",       #Hyperbolic Secant
    "invgamma",        #Inverse Gamma
    "invweibull",      #Inverse Weibull
    "johnsonsb",       #Johnson SB
    "johnsonsu",       #Johnson SU
    "laplace",         #Laplace
    "logistic",        #Logistic
    "loggamma",        #Log-Gamma
    "loglaplace",      #Log-Laplace (Log Double Exponential)
    "lognorm",         #Log-Normal
    "lomax",           #Lomax (Pareto of the second kind)
    "maxwell",         #Maxwell
    "mielke",          #Mielke's Beta-Kappa
    "nakagami",        #Nakagami
    "ncx2",            #Non-central chi-squared
    "ncf",             #Non-central F
    "nct",             #Non-central Student's T
    "pareto",          #Pareto
    "powerlaw",        #Power-function
    "powerlognorm",    #Power log normal
    "powernorm",       #Power normal
    "rdist",           #R distribution
    "reciprocal",      #Reciprocal
    "rayleigh",        #Rayleigh
    "rice",            #Rice
    "recipinvgauss",   #Reciprocal Inverse Gaussian
    "semicircular",    #Semicircular
    "t",               #Student's T
    "triang",          #Triangular
    "truncexpon",      #Truncated Exponential
    "truncnorm",       #Truncated Normal
    "tukeylambda",     #Tukey-Lambda
    "uniform",         #Uniform
    "vonmises",        #Von-Mises (Circular)
    "wald",            #Wald
    "weibull_min",     #Minimum Weibull (see Frechet)
    "weibull_max",     #Maximum Weibull (see Frechet)
    "wrapcauchy",      #Wrapped Cauchy
    "ksone",           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign"]       #Kolmogorov-Smirnov two-sided test for Large

    reviewers_df = Features.compute_features()
    print("Calculate Rating Deviation Distribution")
    reviewers_df['rating_deviation_distribution']=calculate_rating_deviation_distribution(reviewers_df,cdfs)
    print("Calculate Avg Helpfulness Distribution")
    reviewers_df['avg_helpfulness_distribution']=calculate_average_helpfulness_distribution(reviewers_df,cdfs)
    print("Calculate Burst Ratio Distribution")
    reviewers_df['burst_ratio_distribution']=calculate_burst_ratio_distribution(reviewers_df,cdfs)
    print("Calculate Number of Reviews Distribution")
    reviewers_df['num_of_reviews_distribution']=calculate_number_reviews_distribution(reviewers_df,cdfs)
    print("Calculate Similarity Index Distribution")
    #reviewers_df['similarity_index_distribution']=calculate_text_similarity_distribution(reviewers_df,cdfs)
    print("Calculate Common Products Distribution")
    #reviewers_df['common_products_distribution']=calculate_product_count_distribution(reviewers_df,cdfs)
    
    