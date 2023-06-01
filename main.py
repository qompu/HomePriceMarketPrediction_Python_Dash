#################################################################
#                                                               #
# DATA SCIENCE WITH PYTHON: AN APPLICATION THAT PROVIDES        #
# EARLY FORECAST SIGNALS FOR REAL ESTATE MARKETS IN THE U.S.    #
#                                                               #
# By Manuel Tamayo                                              #
#################################################################
# main.py: The entire program is contained in this file because
# the Dash app accesses the Pandas dataframe during graph updates.
# Future versions may include a global Pandas dataframe which can
# be accessed by Dash from any file in the project.
#  The Web app will be running at the link that appears at the
#  terminal window after running this file successfully.
#  The URL should be http://127.0.0.1:8050/
#################### REQUIRED PACKAGES ###########################
# Import pandas library. Reference: https://pandas.pydata.org/docs/
import pandas as pd
# Dash apps. Reference https://dash.gallery/Portal/
import dash
from dash import html
# Dash Core Components: components for interactive user interfaces
# https://dash.plotly.com/dash-core-components
from dash import dcc
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import numpy as np
# Statsmodels Lowess reference: https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
from statsmodels.nonparametric.smoothers_lowess import lowess
from dash.dependencies import Input, Output, State
import matplotlib.dates as dates

######################## READ AND PREPARE CSV FILE ############################

# Function reads data/HPI_master.csv prepares the index,
# drops all rows that are not monthly data,
# sets the location column name and data,
# saves results to data/HPI_master_indexed_dparsed.csv
def upload_data():
    try:
        # Read csv add date column from yr and period columns to dataframe
        df_i = pd.read_csv('data/HPI_master.csv', parse_dates={"date": ['yr', 'period']}, keep_date_col=True)
    except FileNotFoundError:
        print("File not found.")
    except pd.errors.EmptyDataError:
        print("No data in csv file")
    except pd.errors.ParserError:
        print("Parse error")
    except Exception:
        print("Some other Read exception")

    # to_datetime Ref: https://www.geeksforgeeks.org/python-pandas-to_datetime/
        df_i["date"] = pd.to_datetime(df_i["date"])  # parse dates

    # Drop quarterly rows with specific values ref: https://www.statology.org/pandas-drop-rows-with-value/
    # define values
    values = ['quarterly']
    df_i = df_i[df_i.frequency.isin(values) == False]
    # create location column which can be merged with other fields in future releases
    df_i["location"] = df_i["place_name"]  # + " - " + df["place_id"]  # use "+" to combine location info if needed
    try:
        df_i.to_csv(r'data/HPI_master_indexed_dparsed.csv', index=True, header=True)  # save processed dataframe to drive
    except FileNotFoundError:
        print("File not found.")
    except pd.errors.EmptyDataError:
        print("No data in csv file")
    except pd.errors.ParserError:
        print("Parse error")
    except Exception:
        print("Some other Write exception")

upload_data()  # Read and prepare csv file

# Load data from indexed csv file
try:
    df = pd.read_csv('data/HPI_master_indexed_dparsed.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("File not found.")
except pd.errors.EmptyDataError:
    print("No data in csv file")
except pd.errors.ParserError:
    print("Parse error")
except Exception:
    print("Some other Read exception")

# convert 'date' column to datetime format
# Reference https://github.com/codebasics/py/blob/master/pandas/17_ts_to_date_time/pandas_ts_to_date_time.ipynb
# Documentation https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
df['date'] = pd.to_datetime(df['date'])

# Drop Rows with NaN Values in Pandas DataFrame. Reference: https://datatofish.com/dropna/
df.dropna()

########################## SET INITIAL DATES FOR THE GRAPH  #########################
# Get initial date from dataframe at position 0 in string format:  mystart_date
# Get max date for graph from max index value date in string format: myend_date
# Set new index column to datetime64 for slicing to work
# Answer 2: https://stackoverflow.com/questions/49554491/not-supported-between-instances-of-datetime-date-and-str
s = pd.to_datetime(df['date'], unit='D')
assert str(s.dtype) == 'datetime64[ns]'
df.index = s  # set new index

df['date_txt'] = pd.to_datetime(
    df['date']).dt.date  # convert to string keep only yyy-mm-dd date drop time 00:00:00 in column date

# get first value from column date_txt
# https://www.codegrepper.com/code-examples/python/get+first+value+from+column+pandas
# get the first and last date as string https://note.nkmk.me/en/python-pandas-head-tail/
mystart_date = str(df['date_txt'].iloc[0])

# find index max https://pandas.pydata.org/docs/reference/api/pandas.Index.max.html
idx = pd.Index(df['date_txt'])

myend_date = str(idx.max())

############################## CREATE DASHBOARD ###########################
###### Create the modal info window
# Reference: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/modal/
modal = html.Div(
    [
        dbc.Button("INFO", id="open", style={'margin-left': '20px'}),
        dbc.Modal(
            [
                dbc.ModalHeader("Housing Price Index, Trends and Forecasts."),
                dbc.ModalBody([
                                html.P("Forecast: Based on the Lowess regression model (frac value 0.2) for "
                                       "scatter plot forecast. Only the current point is used for the forecast "
                                       "because the entire curve is redrawn when the parameters change."),
                                html.P("MACD: Moving average convergence/ divergence method. A curve is drawn "
                                       "from the difference of 2 moving averages. A moving average of the macd "
                                       "is the signal. The crossovers of these lines indicate a possible change "
                                       "in momentum and trend. The MACD settings are: Moving averages 12 and 24 "
                                       "with a signal moving average of 9."),
                                html.P("Navigation: Select time frame in the pop-up selector or type it in the "
                                       "input box. Select the geographic location for the data in the drop down menu. "
                                       "Drag the cross hair icon to focus. Double click to reset. "),
                                html.P("Note: The pointer may need to be double clicked to view the entire y axis "
                                       "scale after time frame change or zoom change. "),
                                html.P("Data provided courtesy Federal Housing Finance Agency (FHFA) www.fhfa.gov. "),
                                html.P("This product uses FHFA Data but is neither endorsed nor certified by FHFA."),
                                html.P("Programmed by Manuel Tamayo for Capstone project class. "),

                              ]),

                dbc.ModalFooter(
                    dbc.Button("CLOSE", id="close", className="ml-auto")
                ),
            ],
            id="modal",
            is_open=False,  # True, False
            size="xl",  # "sm", "lg", "xl"
            backdrop=True,  # True, False use Static for modal to close by clicking on backdrop
            scrollable=True,  # False or True for scrolling text
            centered=True,  # True, False
            fade=True,  # True, False
            style={"color": "#333333"}  # set text color style inside the modal
        ),
    ]
)
##### INITIALIZE THE DASH APP #####
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # Reference: https://bootswatch.com/default/
app.config.suppress_callback_exceptions = True  # suppress callback exception output

########################### BUILD APP LAYOUT ############################
# Reference: https://www.statworx.com/en/blog/how-to-build-a-dashboard-in-python-plotly-dash-step-by-step-tutorial/
app.layout = html.Div(
    children=[
        # Add html to the app
        html.Br(),

        modal,  # Display the Info modal Button

        html.Br(),

        html.Div(className='row',
                 children=[
                     html.Div(className='four columns div-user-controls',
                              children=[
                                  html.H2('Housing Price Index Dashboard'),
                                  html.P('Select a time range below.'),
                                  # DatePickerRange html layout. Ref.:https://www.youtube.com/watch?v=5uwxoxaPD8M
                                  # Documentation: https://dash.plotly.com/dash-core-components/datepickerrange
                                  dcc.DatePickerRange(
                                      id='date-picker-range1',  # ID for callback

                                      day_size=39,  # size of calendar image. Default is 39

                                      start_date_placeholder_text="Start Period",
                                      end_date_placeholder_text="End Period",
                                      calendar_orientation='horizontal',
                                      with_full_screen_portal=False,
                                      # True will open in a full screen overlay portal
                                      with_portal=True,  # True will open in a full screen overlay portal
                                      first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
                                      reopen_calendar_on_clear=True,
                                      is_RTL=False,  # True or False for direction of calendar
                                      clearable=False,  # whether or not the user can clear the dropdown
                                      number_of_months_shown=2,  # number of months shown when calendar is open
                                      min_date_allowed=mystart_date,
                                      # minimum date allowed on the DatePickerRange component
                                      max_date_allowed=myend_date,
                                      # maximum date allowed on the DatePickerRange component
                                      initial_visible_month=mystart_date,
                                      # the month initially presented when the user opens the calendar
                                      start_date=mystart_date,

                                      end_date=myend_date,
                                      display_format='MMM DD, YYYY',
                                      # how selected dates are displayed in the DatePickerRange component.
                                      month_format='MMMM, YYYY',
                                      # how calendar headers are displayed when the calendar is opened.
                                      minimum_nights=30,  # minimum number of days between start and end date

                                      persistence=True,
                                      persisted_props=['start_date'],
                                      persistence_type='session',  # session, local, or memory. Default is 'local'

                                      updatemode='singledate'
                                      # singledate or bothdates. Determines when callback is triggered
                                  ),

                                  html.P('Select place or region from the dropdown list below.'),
                                  html.Div(
                                      className='div-for-dropdown',
                                      children=[
                                          # Dropdown menu documentation https://dash.plotly.com/dash-core-components/dropdown
                                          dcc.Dropdown(id='geo-dropdown',
                                                       options=[{'label': i, 'value': i}
                                                                for i in df['location'].unique()],  # get location from dataframe
                                                       value='United States',  # Default selection from 'location' column
                                                       multi=False,
                                                       clearable=False,  # X clears selection
                                                       searchable=True,  # Search feature enabled
                                                       style={'backgroundColor': '#1E1E1E'},
                                                       className='geo-dropdown'
                                                       ),

                                      ],
                                      style={'color': '#1E1E1E'}),


                              ]
                              ),
                     html.Div(className='eight columns div-for-charts bg-grey',
                              children=[
                                  # Graph header
                                  html.H3("Housing Price Index, Trends and Forecasts.", style={'textAlign': 'center'}),
                                  html.P("This product uses FHFA Data but is neither endorsed nor certified by FHFA.", style={'textAlign': 'center'}),
                                  dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                              ])
                 ])
    ]

)

# Set up the callback function for the INFO modal
# References:
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/modal/
# https://python.plainenglish.io/how-to-create-a-model-window-in-dash-4ab1c8e234d3
# https://www.youtube.com/watch?v=X3OuhqS8ueM
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# Set up the callback function for the graphs
@app.callback(  # Callback header
    Output(component_id='timeseries', component_property='figure'),

    [Input(component_id='date-picker-range1', component_property='start_date'),
     Input(component_id='date-picker-range1', component_property='end_date'),
     Input(component_id='geo-dropdown', component_property='value'),
     ]
)
# function updates graph
def update_graph(start_date, end_date, value):
    dateslice = df.loc[str(start_date):str(end_date)]

    filtered_hpi = dateslice[dateslice['place_name'] == value]  # filtered dataframe

    # Create subplots. Reference: https://plotly.com/python/subplots/#customizing-subplot-axes
    # Update subplots. Reference: https://plotly.com/python/creating-and-updating-figures/

    ################## COMPUTE MOVING AVERAGES, LOWESS REGRESSION, AND MACD (settings: 12, 24, 9)
    # add moving averages with periods 12 and 24 to data frame
    # RefL https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

    # compute lowess regression forecast on the HPI seasonally adjusted dataseries
    # First construct numerical arrays from columns "date" and "index_sa"
    # set x1 axis as numpy array
    x1 = filtered_hpi['date'].to_numpy()
    # use matplotlib.dates to convert date column to numpy array of numbers
    x1_dates = dates.date2num(x1)
    # set y1 axis as numpy array of index_sa values
    y1 = filtered_hpi['index_sa'].to_numpy()
    # Compute a lowess forecast Frac value of 0.2 on the data.
    # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    lowess_forecast1 = lowess(exog=x1_dates, endog=y1, frac=0.2)  # creates a series not a dataframe
    # print(lowess_forecast1)
    filtered_hpi.insert(0, "forecast", lowess_forecast1[:, 1], allow_duplicates=True)  # Merge Lowess to dataframe

    ma12 = filtered_hpi.index_sa.rolling(12).mean()
    filtered_hpi["MA 12"] = ma12
    ma24 = filtered_hpi.index_sa.rolling(24).mean()
    filtered_hpi["MA 24"] = ma24
    macd = ma12 - ma24
    filtered_hpi["MACD"] = macd
    signal = filtered_hpi.MACD.rolling(9).mean()
    filtered_hpi["signal"] = signal

    #################### FUNCTIONS FOR FUTURE EXPANSION  ######################
    # Compute moving average on numpy array
    # Reference https://www.delftstack.com/howto/python/moving-average-python/
    def moving_average(np_array, ma_period):
        return np.convolve(np_array, np.ones(ma_period), 'valid') / ma_period


################################### CREATE SUBPLOTS #######################################
    # Create subplots. Reference: https://plotly.com/python/subplots/#customizing-subplot-axes
    # Update subplots. Reference: https://plotly.com/python/creating-and-updating-figures/

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Housing Price Index (seasonally adjusted) for: {value}", "MACD Trend Momentum Indicator. Settings: MA 12, 24, Signal 9."))
    # f string contains embedded expression
    ##### TOP GRAPH #####
    # Add new traces. Reference: https://www.kite.com/python/docs/plotly.graph_objs.Figure.add_scatter
    fig.add_scatter(y=filtered_hpi['index_sa'], x=filtered_hpi['date'],
                    line=dict(color='Blue'),
                    name="INDEX", row=1, col=1)

    fig.add_scatter(y=filtered_hpi['forecast'], x=filtered_hpi['date'],
                    line=dict(color='Orange'),
                    name="forecast", row=1, col=1)

    fig.add_scatter(y=filtered_hpi['MA 12'], x=filtered_hpi['date'],
                    line=dict(color='Teal'),
                    name="MA 12", row=1, col=1,
                    line_width=1),
    fig.add_scatter(y=filtered_hpi['MA 24'], x=filtered_hpi['date'],
                    line=dict(color='RoyalBlue'),
                    name="MA 24", row=1, col=1,
                    line_width=1),

    ##### BOTTOM GRAPH #####
    fig.add_scatter(y=filtered_hpi['MACD'], x=filtered_hpi['date'],
                    line=dict(color='Green'),
                    name="MACD", row=2, col=1)

    fig.add_scatter(y=filtered_hpi['signal'], x=filtered_hpi['date'],
                    line=dict(color='Firebrick'),
                    name="signal", row=2, col=1)

    fig.update_layout(plot_bgcolor="Gainsboro")

    return fig

############################ RUN APP SERVER ##########################
if __name__ == '__main__':
    app.run_server(debug=False)
# to run the app in a production environment use gunicorn

