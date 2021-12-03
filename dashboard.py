import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import shap
from textwrap import wrap
import seaborn as sns
import matplotlib.pyplot as plt


# loading some of the necessary files / data
X_test_notscaled = pd.read_csv('X_test.csv')

X_test_notscaled_renamed = X_test_notscaled.copy()
X_test_notscaled_renamed.columns = [col.split('_//', 1)[0] for col in X_test_notscaled.columns]

model = joblib.load('Best_model.pkl')

shap_values = joblib.load('shap_values.pkl')
shap_col_order = joblib.load('col_order.pkl')


# Defining some variables
datatypes = ['Top5 increasing the client\'s score',
             'Top5 decreasing the client\'s score',
             'General information',
             'External source scores',
             'Previous applications from other banks',
             'Previous applications from our bank',
             'Previous credits from our bank',
             'Credit Card information']

thresh = 0.35

SHAP_base_value = -0.5224664169653608


# Function to predict probabilities not to repay a loan and classify accordingly, applied to a dataframe
@st.cache
def predict_class(dataframe):
    pred_class = np.where(model.predict_proba(dataframe)[:, 1] < thresh,
                          'loan granted',
                          'loan not granted')
    return pred_class


# function to enable a shap plot to be displayed through streamlit:
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


###########
# Sidebar #
###########
logo = Image.open('logo.jpg')
st.sidebar.image(logo)
st.sidebar.write(''' ''')
IDclient = st.sidebar.selectbox('Client ID', X_test_notscaled['SK_ID_CURR'])
datatype = st.sidebar.selectbox('Data Type to display', datatypes)


#############
# Main part #
#############

st.write("""# Loan scoring""")


# ########### Results section ############
st.header('*Results*')
col1, col2, col3 = st.columns(3)


# ##### col1 : model's prediction for the selected client
data_ID = X_test_notscaled[X_test_notscaled['SK_ID_CURR'] == IDclient]  # define input data for the model
pred_proba = model.predict_proba(data_ID)[:, 1]  # make the prediction
col1.metric(label="Probability not to repay the loan",
            value=round(pred_proba[0], 2),
            delta=round(-0.35+pred_proba[0], 2),
            delta_color="inverse")


# ##### col2 : theshold
col2.metric(label="Threshold for granting a loan",
            value=thresh,
            delta=None)


# ##### col3 : corresponding outcome (loan granted or not)
if pred_proba[0] < thresh:
    outcome = '<p style="color:#10AF1D; font-size: 20px;">Congratulations, you\'ve been granted this loan!</p>'
else:
    outcome = '<p style="color:#D40D0A; font-size: 20px;">Sorry, we are not able to grant you this loan...</p>'
col3.write('###### Outcome')
col3.markdown(outcome, unsafe_allow_html=True)


# ##### Below the 3 columns, display the corresponding shap force plot
st.write(''' ''')
st.write('##### -------------------------------------- '
         'Why such a probability: '
         '--------------------------------------')

# SHAP_base_value --> defined above
# shap_values --> loaded above
# shap data :
shap_data = X_test_notscaled.set_index('SK_ID_CURR')  # we want the non-scaled data to be displayed on the plot
shap_data.columns = [lab.split('_//', 1)[0] for lab in shap_data.columns]  # rename columns with short names
shap_data = shap_data[shap_col_order]  # re-order columns to fit with SHAP analysis
#                                (SHAP analysis performed on scaled data - scaling re-orders columns)

# defining the row index corresponding to the client's ID
client_row = X_test_notscaled[X_test_notscaled['SK_ID_CURR'] == IDclient].index[0]

# Make and display the force plot
shap_plot = shap.force_plot(SHAP_base_value,
                            shap_values[client_row],
                            shap_data.loc[IDclient, :],
                            link='logit',
                            plot_cmap=["#D40D0A", "#10AF1D"])
st_shap(shap_plot, 120)


# ########### Client's data section ############
st.write('''***''')
st.header('*Client\'s data*')

# reduce data to those of the client
df = X_test_notscaled_renamed[X_test_notscaled_renamed['SK_ID_CURR'] == IDclient]

# display them according to datatype selected
if datatype == 'Top5 increasing the client\'s score':
    highest_var_idx = np.argsort(shap_values[client_row])[-5:][::-1]
    highest_var = [shap_data.columns[idx] for idx in highest_var_idx]
    df_ = df[highest_var]
    df__ = df_.copy()
    st.write('''*(left) increasing the most  >>>  (right) increasing the least*''')
if datatype == 'Top5 decreasing the client\'s score':
    lowest_var_idx = np.argsort(shap_values[client_row])[::-1][-5:][::-1]
    lowest_var = [shap_data.columns[idx] for idx in lowest_var_idx]
    df_ = df[lowest_var]
    df__ = df_.copy()
    st.write('''*(left) decreasing the most  >>>  (right) decreasing the least*''')
if datatype == 'General information':
    df_ = df[['NEW_AGE',
              'NEW_DAYS_EMPLOYED_PERC',
              'AMT_CREDIT',
              'NEW_LVR',
              'NEW_PROD_CRED_SALARY']]
    df__ = df_.copy()
    df__.columns = ['Age',
                   'Employment relative duration',
                   'Loan amount',
                   'Loan to good Value Ratio',
                   '(Good value - Loan amount) / Income']
if datatype == 'External source scores':
    df_ = df[['EXT_SOURCE_1',
              'EXT_SOURCE_2',
              'EXT_SOURCE_3',
              'NEW_EXT_SOURCE_MEAN',
              'NEW_EXT_SOURCE_STD']]
    df__ = df_.copy()
    df__.columns = ['Score banque1',
                   'Score banque2',
                   'Score banque3',
                   'Mean',
                   'Standard deviation']
if datatype == 'Previous applications from other banks':
    df_ = df[['BURO_DAYS_CREDIT__min',
              'BURO_DAYS_CREDIT__max',
              'BURO_DAYS_CREDIT_ENDDATE__min',
              'BURO_DAYS_CREDIT_ENDDATE__max',
              'BURO_CREDIT_ACTIVE_Active__count__max',
              'BURO_CREDIT_ACTIVE_Closed__count__max']]
    df__ = df_.copy()
    df__.columns = ['First application (days ago)',
                   'Last application (days ago)',
                   'Shortest loan remaining duration',
                   'Longest Loan remaining duration',
                   'Nb of active credits',
                   'Nb of closed credits']
if datatype == 'Previous applications from our bank':
    df_ = df[['PREV_AMT_ANNUITY__min',
              'PREV_AMT_ANNUITY__max',
              'PREV_CNT_PAYMENT__max',
              'PREV_CNT_PAYMENT__mean',
              'PREV_APP_SUCCESS_RATE__mean',
              'PREV_APP_LVR__min',
              'PREV_APP_LVR__max',
              'prevAPPROVED_APP_LVR__max',
              'prevREFUSED_APP_LVR__min']]
    df__ = df_.copy()
    df__.columns = ['Previous minimal annuity',
                   'Previous maximal annuity',
                   'Longest term of previous credits',
                   'Mean term of previous credit',
                   'Mean rate of granted credits',
                   'Minimal previous loan to good ratio',
                   'Maximal previous loan to good ratio',
                   'Highest previous loan to good ratio approved',
                   'Lowest previous loan to good ratio refused']
if datatype == 'Previous credits from our bank':
    df_ = df[['POS_COUNT',
              'INSTAL_PAYMENT_DIFF__mean',
              'INSTAL_PAYMENT_DIFF__sum']]
    df__ = df_.copy()
    df__.columns = ['Nb of previous loans',
                   'Mean diff. of installment due vs installment paid',
                   'Total diff. of installment due vs installment paid']
if datatype == 'Credit Card information':
    df_ = df[['CC_COUNT',
              'CC_AMT_DRAWINGS_CURRENT__mean',
              'CC_AMT_DRAWINGS_CURRENT__std',
              'CC_AMT_TOTAL_RECEIVABLE__mean',
              'CC_AMT_TOTAL_RECEIVABLE__std',
              'CC_CNT_DRAWINGS_CURRENT__mean',
              'CC_CNT_DRAWINGS_CURRENT__std']]
    df__ = df_.copy()
    df__.columns = ['Nb of credit card lines',
                   'Mean monthly drawing amount',
                   'Standard deviation monthly drawing amount',
                   'Mean total amount receivable',
                   'Stadard deviation total amount receivable',
                   'Mean nb of monthly drawings',
                   'Standard deviation of nb of monthly drawings']
df__ = df__.set_index([pd.Index([IDclient])])
st.write(df__)
st.write('''_**Note**: <NA> indicates that data are not available / have not been provided_''')


# ########### Comparison section ############
st.write('''***''')
st.header('*Comparison with other clients*')

# Predict classes according to threshold for all clients
pred_class = predict_class(X_test_notscaled)

# build df with data to compare according to datatype selected
data_comp = X_test_notscaled_renamed[df_.columns]

# add predicted classes (pred_proba) to this df
data_comp = pd.concat([data_comp, pd.Series(pred_class, name='pred_proba')], axis=1)

# replace pred_proba value by 'client(ID)' for the selected client
data_comp.loc[client_row, 'pred_proba'] = 'Client ({})'.format(IDclient)

# Build the plot
total_cols = data_comp.shape[1]-1
total_rows = 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(5 * total_cols, 5 * total_rows), constrained_layout=True)
for i, var in enumerate(df_.columns):
    pos = i % total_cols
    plot = sns.boxplot(x='pred_proba', y=var, data=data_comp,
                       ax=axs[pos],
                       order=['loan granted',
                              'Client ({})'.format(IDclient),
                              'loan not granted'],
                       palette={'loan granted': "#10AF1D",
                                'Client ({})'.format(IDclient): 'black',
                                'loan not granted': "#D40D0A"},
                       # whis=0, # for no whiskers at all
                       whis=(0, 100),  # whiskers cover the whole range of data
                       showfliers=False,
                       showmeans=True,
                       meanprops={'marker': 'o',
                                  'markersize': 10,
                                  'markeredgecolor': 'black',
                                  'markerfacecolor': 'white'}
                       )
    plot.set_title('\n'.join(wrap(df__.columns[i], 20)),
                   fontsize=26, fontweight='bold')
    plot.set_xlabel('')
    plot.set_ylabel('')
    plot.set_xticklabels(['Loan\ngranted',
                          'Client\n({})'.format(IDclient),
                          'Loan\nnot\ngranted'],
                         fontsize=20)

# display the plot
st.write(fig)
st.write('''_**Notes**:_
_boxes extend from the first to third quartiles of the data of each group (loan granted vs not granted);_
_the horizontal line represents the median value;_
_the white dot represents the mean value;_
_the whiskers cover the whole range of data for each group._
''')
