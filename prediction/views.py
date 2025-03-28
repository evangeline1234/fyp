from django.shortcuts import render
import os
import pandas as pd
import pickle
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionRequestSerializer

# Create your views here.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)  # Load the XGBoost model
except Exception as e:
    model = None  # Handle loading failure


def prepare_input(carparkcode, datetime):
    public_holidays = ['2024-08-09', '2024-10-31', '2024-12-24', '2024-12-25', '2025-01-01', '2025-01-29', '2025-01-30', '2025-03-31', '2025-04-18', '2025-05-01', '2025-05-12', '2025-06-07']
    food_area_carparks = ['U0042', 'U0036', 'L0078', 'J0117', 'T0103', 'T0140', 'L0117', 'A0046', 'L0064', 'L0107', 'L0116', 'N0012', 'Q0006', 'B0087', 'P0111', 'S0150', 'H0003', 'J0054', 'B0031', 'S0049', 'V0007', 'D0026', 'H0057', 'H0015', 'K0111', 'M0040', 'M0059', 'C0133', 'H0024', 'O0028', 'S0171', 'H0022', 'P0096', 'C0148', 'A0021', 'P0117', 'D0006', 'N0006', 'Y0019', 'A0007', 'M0078', 'S0112', 'H0011', 'W0055', 'P0054', 'P0094', 'P0106', 'A0024', 'J0055', 'B0088', 'J0092', 'S0020', 'S0055', 'S0166', 'J0122', 'L0104', 'E0023', 'E0024', 'E0027', 'J0054', 'S0049']
    parks_carparks = ['P0048', 'T0017', 'B0063', 'P0109', 'E0023', 'E0024', 'E0027', 'W0029', 'C0162', 'K0082']
    recreational_carparks = ['U0042', 'U0036', 'L0078', 'J0117', 'T0103', 'C0119', 'T0140', 'T0129', 'L0124', 'L0125', 'L0117', 'A0046', 'L0116', 'L0064', 'L0107', 'L0123', 'Q0006', 'N0013', 'B0087', 'P0111', 'S0150', 'P0075', 'K0039', 'K0037', 'H0003', 'J0054', 'B0031', 'S0049', 'V0007', 'K0121', 'D0026', 'H0057', 'H0015', 'K0111', 'M0040', 'M0059', 'P0096', 'A0011', 'A0021', 'P0117', 'D0006','N0006', 'P0096', 'H0022', 'O0028', 'H0024', 'C0133']
    carparkNo = ['A0007' 'A0011' 'A0017' 'A0021' 'A0024' 'A0035' 'A0046' 'B0031' 'B0032'
                'B0063' 'B0087' 'B0088' 'B0098' 'C0119' 'C0133' 'C0148' 'C0162' 'D0006'
                'D0026' 'D0028' 'E0023' 'E0024' 'E0027' 'H0003' 'H0011' 'H0015' 'H0022'
                'H0024' 'H0057' 'J0017' 'J0054' 'J0055' 'J0092' 'J0100' 'J0122' 'K0037'
                'K0039' 'K0082' 'K0111' 'K0121' 'L0064' 'L0078' 'L0104' 'L0107' 'L0116'
                'L0117' 'L0123' 'L0124' 'L0125' 'M0040' 'M0059' 'M0076' 'M0078' 'M0084'
                'M0088' 'N0006' 'N0012' 'N0013' 'O0028' 'P0013' 'P0033' 'P0048' 'P0054'
                'P0075' 'P0093' 'P0094' 'P0096' 'P0106' 'P0109' 'P0111' 'P0113' 'P0117'
                'Q0006' 'Q0008' 'S0020' 'S0049' 'S0055' 'S0106' 'S0108' 'S0112' 'S0150'
                'S0166' 'S0171' 'T0017' 'T0103' 'T0129' 'T0140' 'U0036' 'U0042' 'V0007'
                'W0029' 'W0055' 'Y0019'
    ]
    carparkNo = carparkcode
    hour = datetime.hour
    minute = datetime.minute
    dayofweek = datetime.weekday()
    is_weekend = 1 if dayofweek >= 5 else 0  # 5 and 6 correspond to Saturday and Sunday
    is_PH = 1 if datetime in public_holidays else 0  # Check if it's a public holiday
    lunch_dinner_hours = 1 if (hour >= 12 and hour <= 15) or (hour >= 18 and hour <= 21) else 0  # Check if it's lunch/dinner hours
    office_hours = 1 if hour >= 9 and hour <= 18 else 0  # Check if it's office hours
    near_food_area = 1 if carparkcode in food_area_carparks else 0  # Check if it's a food area carpark
    near_parks = 1 if carparkcode in parks_carparks else 0  # Check if it's a parks carpark
    near_recreational_activities = 1 if carparkcode in recreational_carparks else 0  # Check if it's a recreational carpark

    input_data = [[carparkNo, hour, minute, dayofweek, is_weekend, is_PH, lunch_dinner_hours, office_hours, near_food_area, near_parks, near_recreational_activities]]
    input_data = pd.DataFrame(input_data, columns=['carparkNo', 'hour', 'minute', 'dayofweek', 'is_weekend', 'is_PH', 'lunch_dinner_hours', 'office_hours', 'near_food_area', 'near_parks', 'near_recreational_activities'])
    input_data['carparkNo'] = input_data['carparkNo'].astype('category')
    X = os.path.join(os.path.dirname(__file__), "X.csv")
    X = pd.read_csv(X)
    X = pd.DataFrame(X)
    X['carparkNo'] = X['carparkNo'].astype('category')
    
    filtered_df = X[(X['hour'] == input_data['hour'].iloc[0]) & (X['minute'] == input_data['minute'].iloc[0]) & (X['dayofweek'] == input_data['dayofweek'].iloc[0]) &
    (X['is_weekend'] == input_data['is_weekend'].iloc[0]) & (X['is_PH'] == input_data['is_PH'].iloc[0]) & (X['lunch_dinner_hours'] == input_data['lunch_dinner_hours'].iloc[0]) &
    (X['office_hours'] == input_data['office_hours'].iloc[0]) & (X['near_food_area'] == input_data['near_food_area'].iloc[0]) & (X['near_parks'] == input_data['near_parks'].iloc[0]) &
    (X['near_recreational_activities'] == input_data['near_recreational_activities'].iloc[0]) & (X['carparkNo'] == 'A0017')
    ]
    # Take the last 3 matching rows and compute the mean lag_1hr
    lag_1hr_mean = filtered_df['lag_1hr'].tail(3).mean()
    lag_24hr_mean = filtered_df['lag_24hr'].tail(3).mean()
    # Use mean of last 3 lagged values from original df with same features for the future date
    input_data.loc[0, 'lag_1hr'] = lag_1hr_mean
    input_data.loc[0, 'lag_24hr'] = lag_24hr_mean
    
    return input_data



class PredictAvailabilityView(APIView):
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if serializer.is_valid():
            carparkcode = serializer.validated_data['carparkcode']
            datetime = serializer.validated_data['datetime']

            if model is None:
                return Response({"error": "Prediction model could not be loaded."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Prepare data using the function
            input_data = prepare_input(carparkcode, datetime)

            try:
                predicted_lots = model.predict(input_data)[0]

                return Response({
                    "carparkcode": carparkcode,
                    "datetime": datetime,
                    "predicted_lots": int(predicted_lots)  # Ensure it's an integer
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)