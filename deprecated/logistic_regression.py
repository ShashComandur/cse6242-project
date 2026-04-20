# libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv("model_ready_dataset.csv")

# print(df.head())

# select relevant columns
analysis_cols = [
    'release_speed',                # pitch speed
    'release_pos_x',                # horizontal release position of ball measured in feet from the catcher's perspective
    'release_pos_z',                # vertical release position of ball measured in feet from the catcher's perspective
    'pfx_x',                        # horizontal movement in feet from the catcher's perspective
    'pfx_z',                        # vertical movement in feet from catcher's perspective
    'plate_x',                      # horizontal position of ball when it crosses home plate from catcher's perspecitve
    'plate_z',                      # vertical position of ball when it crosses home plate from catcher's perspective
    'release_spin_rate',            # spin rate of pitch
    'release_extension',            # release extension of pitch in feet
    'spin_axis',                    # spin axis in the 2D X-Z plane in degrees (0-360), where 180 = pure backspin fastball and 0 = pure topspin curveball
    'event_group'                   # outcome variable
]

# update df
df = df[analysis_cols]

# separate predictors and outcomes
y = df['event_group']
X = df.drop(columns=['event_group'])

# scaling predictor data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

# split intro train / test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

# build model
model = LogisticRegression(solver='saga', max_iter=50)      # may need multi_class='multinomial' depending on version of scikit
model.fit(X_train, y_train)

# coefficients for further interpretation
coef_table = pd.DataFrame(
    model.coef_, 
    columns=X.columns, 
    index=model.classes_
)

# view all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(coef_table)
