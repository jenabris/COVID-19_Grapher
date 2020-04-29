import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from scipy import interpolate
import os
import datetime as dt

# import R packages
from rpy2.robjects.packages import importr
from scipy.integrate import odeint

#Create python objects representing the R packages.
base = importr('base')
ee = importr('EpiEstim')
lyr = importr('dplyr')
lub = importr('lubridate')


def country_plots(cname, pop, lag, dtype, sdstart, incu):
    #Convert strings to datetime objects.
    sdstart = dt.datetime.strptime(dt.datetime.strptime(sdstart, "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

    # Making figure.
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 20))
    fig.autofmt_xdate()
    fig.tight_layout(pad=1.0)

    #Setting up country database.
    cdata = makedatabase(dtype)
    country_vec = extractcountry(cdata, cname)
    country_vec = dropmetacolumns(country_vec)

    #Plotting first plot: deaths/confirmed cases.
    countrylogplot(country_vec, sdstart, incu, ax1)

    #Plotting second plot: SIR model.
    t_out, s_out, i_out, r_out, rx_out, iover_out, result, plc_out, pcinf_out, diffinf_out = calculate_SIR(country_vec, lag, pop, cname)
    plot_SIR(ax2, t_out, s_out, i_out, r_out, rx_out, iover_out, pop, cname)

    #Plotting third plot: R0 over time.
    plot_P0(ax3, result, cname, lag, plc_out, pop, pcinf_out, diffinf_out)

    #Show plot
    plt.show()


#Creates and formats country data.
def makedatabase(whichdata):
    #Setting the current date.
    today = dt.datetime.now().date()
    #Determine which data to read-in.
    if whichdata == "confirmed cases":
        try:
            filetime = dt.datetime.fromtimestamp(os.path.getctime('global_test'))
        except:
            filetime = dt.datetime.now() - dt.timedelta(1)
        if filetime.date() == today:
            df = pd.read_pickle('global_test')
        else:
            urlglobal = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
            # Because of error_bad_lines the returned database may not contain all countries, exception should be caught downstream.
            df = pd.read_csv(urlglobal, error_bad_lines=False)
            df.to_pickle('global_test')

    elif whichdata == "deaths":
        try:
            filetime = dt.datetime.fromtimestamp(os.path.getctime('global_death'))
        except:
            filetime = dt.datetime.now() - dt.timedelta(1)
        if filetime.date() == today:
            df = pd.read_pickle('global_death')
        else:
            urlglobal = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
            df = pd.read_csv(urlglobal, error_bad_lines=False)
            df.to_pickle('global_death')

    else:
        print("ERROR: Wrong datatype name specified for the variable data_type.")

    # Dropping metadata columns.
    #df.drop(df.columns[np.r_[0:4]], inplace=True, axis=1)

    # Transforming dataframe dates to yyyy-mm-dd.
    for x in df.columns[4:]:
        df.rename({str(x): dt.datetime.strptime(dt.datetime.strptime(str(x), "%m/%d/%y").strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")}, inplace=True, axis="columns")

    return df

def dropmetacolumns(df):
    df.drop(df.columns[np.r_[0:4]], inplace=True, axis=1)
    return df

def extractcountry(df, country):
    inf = df.loc[df["Country/Region"] == country, :]
    return inf

def addincubation(date, incu):
    return date + dt.timedelta(incu)

def calculate_SIR(country, psh, pppl, name):
    pr0, plc, pbc = calculate_R0(country, name, lord=False, shf=psh)
    pr0t, plct, pbct = calculate_R0(country, name, lord=True, shf=0)
    presult = pd.concat([pr0, pr0t], join='outer', axis=1)
    plc = plc * 1281
    pbc = pbc * 1281
    pcinf = plc / (pppl * 1000000) * 100
    diffinf = pcinf - (pbc / (pppl * 1000000) * 100)
    s, i, r, rx, t, iover = ode(np.mean(pr0[-7:]), pppl, pppl * (pcinf / 100), pppl * (diffinf / 100))
    return t, s, i, r, rx, iover, presult, plc, pcinf, diffinf

def calculate_R0(cvector, cntname, lord, shf):
    # last case
    lastcase = cvector.iloc[-1].iloc[-1]
    # before last case
    beforecase = cvector.iloc[-1].iloc[-2]
    # convert cumulative data to incidence
    cvector = cvector.diff(axis=1)
    # transpose and reset index (index turned into column)
    cvector = cvector.T.reset_index()
    # name columns
    cvector.columns = ['dates', 'I']
    # custom date-convesrion (variable number of digit problem)
    #cvector['dates'] = cvector['dates'].apply(lambda rowdate: date(rowdate)).copy(deep=True)
    # datetime conversion
    #cvector['dates'] = pd.to_datetime(cvector['dates'])
    # get the indexof last zero to get a continous timeseries
    if not cvector[cvector['I'] == 0].empty:
        idx = cvector[cvector['I'] == 0].index[-1]
        if idx > 30:
            idx = cvector[cvector['I'] == 0].index[-2]
    else:
        idx = 0
    cvector.drop(cvector.index[0:idx], inplace=True)
    # true incidence from deaths
    if not lord:
        cvector['I'] = cvector['I'].apply(lambda inc: d2i(inc)).copy(deep=True)
    # screen for negativ incidence error
    cvector['I'] = cvector['I'].apply(lambda data: neg2zero(data)).copy(deep=True)
    # drop the first always nan row
    cvector = cvector.iloc[1:]

    # allow rpy onversion
    pandas2ri.activate()
    # convert to r dataframe
    rdf = ro.conversion.py2rpy(cvector)
    # import functions
    mutate, estimate = lyr.mutate, ee.estimate_R
    # use Date R object instead of datetimeindex (see format) - rx2 gets the column
    rdfd = mutate(rdf, dates=base.as_Date(rdf.rx2('dates').astype('str'), format="%Y-%m-%d %H:%M:%S"))
    res = estimate(incid=rdfd, method="parametric_si",
                   config=ee.make_config(base.list(mean_si=2.6, std_si=1.5)))  # estimate R
    # convert back to pandas dataFrame
    df_r0 = ro.conversion.rpy2py(res)

    # length of R0 series from today back
    rl = len(df_r0[0]['Mean(R)'])
    # contruct a ts form R0
    idx = cvector['dates'].iloc[-rl:]
    cts = pd.DataFrame(list(df_r0[0]['Mean(R)']), index=cvector['dates'].iloc[-rl:])
    cts[cts[0] > 5] = 5
    cts.index.freq = 'd'
    if not lord:
        cts.columns = [cntname + '_deaths']
        cts.index = cts.index.shift(shf)
    else:
        cts.columns = [cntname + '_cases']
    return cts, lastcase, beforecase

def ode(r0x, n, r0, i0):
    # R0x = 9
    # Total population, N.
    # N = 8750000
    # Initial number of infected and recovered individuals, I0 and R0.
    # I0, R0 = 1000, 1000
    # Everyone else, S0, is susceptible to infection initially.
    s0 = n - i0 - r0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # beta, gamma = 0.12, 1/10
    gamma = 1 / 10
    beta = r0x.tolist()[0] / (1 / gamma)
    # A grid of time points (in days)
    t = np.linspace(0, 2 * 356, 600)

    # The SIR model differential equations.
    def deriv(y, t, nd, betad, gammad):
        sd, idd, rd = y
        dsdt = -betad * sd * idd / nd
        didt = betad * sd * idd / nd - gammad * idd
        drdt = gammad * idd
        return dsdt, didt, drdt

    # Initial conditions vector
    y0 = s0, i0, r0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(n, beta, gamma))
    s, i, r = ret.T
    iover = np.min(np.where(i < np.max(i) * 0.05))


    return s, i, r, r0x, t, iover

def date(usadate):
    datesnums = usadate.split('-', 2)
    if len(datesnums[0]) == 1:
        datesnums[0] = '0' + datesnums[0]
    if len(datesnums[1]) == 1:
        datesnums[1] = '0' + datesnums[1]
    datesnums[2] = '20' + datesnums[2]
    return str(datesnums[2] + '.' + datesnums[0] + '.' + datesnums[1])


def d2i(inc):
    return inc * 1


def neg2zero(data):
    if data < 0:
        return 0
    else:
        return data

def countrylogplot(inf, sdstart, incu, ax):
    #fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.scatter(inf.columns, inf.iloc[0, :].values)
    ax.set_yscale('log')


    if sd:
        ax.axvline(x=sdstart, label="Social distancing begins", color="r")
        ax.axvline(x=addincubation(sdstart, incu), label="Social distancing begins + 1 incubation time", color="g")

    ax.legend(loc=4)
    return

def plot_SIR(ax, t, s, i, r, rx, iover, ppl, country):
    ax.plot(t, s, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, i, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, r, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (millions)')
    ax.set_ylim(0, ppl + 2)
    ax.legend(loc='best', prop={'size': 6})
    ax.title.set_text(
        country + ' R0:' + "{:1.2f}".format(rx.tolist()[0]) + ' - ' + str(iover) + ' days, Dmax:' + "{:1.0f}".format(
            (np.max(i) / 1281) * 1000000) + '(' + "{:1.0f}".format(i.argmax(axis=0)) + ')')

def plot_P0(ax, toplot, pcname, psh, plc, pppl, pcinf, diffinf):
    ax.title.set_text(pcname + ' (lag: ' + str(psh) + ' days) ' + "{:4.2f}".format(plc / 1000000) + '/' + str(
        pppl) + ' (millions) : ' + "{:2.2f}".format(pcinf) + '%' + " ({:2.2f}".format(diffinf) + '%)')
    ax.plot(toplot)
    ax.axhspan(0.5, 1, facecolor='0.5', alpha=0.5)
    ax.axhspan(1.2, 1.4, facecolor='#ff0000', alpha=0.2)

country_name = 'Italy'
lag_days = -11
population = 83.02  # millions
data_type = "deaths" #Either "confirmed cases" or "deaths".
startdate = "2020-03-01" #Startdate for first plot.
enddate = "2020-04-23" #Enddate for first plot.
sd = True #Social distancing.
social_distancing_starts = "2020-03-15" #Begining of the social distancing measures.
incubation_time = 7 #Length of incubation in days. The first plot indicates when the social distancing began and also the timepoint sd + 1 incubation time.
country_plots(country_name, population, lag_days, data_type, social_distancing_starts, incubation_time)