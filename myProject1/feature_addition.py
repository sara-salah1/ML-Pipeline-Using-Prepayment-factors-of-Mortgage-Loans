import pandas as pd
import numpy as np

def calculate_monthly_income(dti, emi):
    dti = dti if dti <= 1 else dti / 100
    if dti == 0:
        monthly_income = emi
    else:
        monthly_income = emi / dti
    return np.int32(monthly_income)

def calculate_emi(principal, monthly_interest_rate, loan_term_months):
    numerator = (1 + monthly_interest_rate) ** loan_term_months
    denominator = numerator - 1
    interest = numerator / denominator
    emi = principal * monthly_interest_rate * interest
    return np.int32(emi)

def get_current_upb(principal, monthly_interest_rate, monthly_installment, payments_made):
    monthly_interest = monthly_interest_rate * principal
    monthly_paid_principal = monthly_installment - monthly_interest
    unpaid_principal = principal - (monthly_paid_principal * payments_made)
    return np.int32(unpaid_principal)

def calculate_prepayment(dti, monthly_income):
    if dti < 40:
        prepayment = monthly_income / 2
    else:
        prepayment = monthly_income * 3 / 4
    return np.int32(prepayment)

def create_features(data):
    data['OrigInterestRate_Monthly'] = np.round((data['OrigInterestRate'] / 12) / 100, 4)
    data['MonthlyInstallment'] = data.apply(
        lambda features: calculate_emi(
            principal=features['OrigUPB'],
            monthly_interest_rate=features['OrigInterestRate_Monthly'],
            loan_term_months=features['OrigLoanTerm']), axis=1)
    data['CurrentUPB'] = data.apply(
        lambda features: get_current_upb(
            monthly_interest_rate=features['OrigInterestRate_Monthly'],
            principal=features['OrigUPB'],
            monthly_installment=features['MonthlyInstallment'],
            payments_made=features['MonthsInRepayment']), axis=1)
    data['MonthlyIncome'] = data.apply(
        lambda features: calculate_monthly_income(
            dti=features['DTI'],
            emi=features['MonthlyInstallment']), axis=1)
    data['Prepayment'] = data.apply(
        lambda features: calculate_prepayment(
            dti=features['DTI'],
            monthly_income=features['MonthlyIncome']), axis=1)
    data['Totalpayment'] = data['MonthlyInstallment'] * data['OrigLoanTerm']
    data['InterestAmount'] = data['Totalpayment'] - data['OrigUPB']
    data['ActualPayments'] = data['MonthlyInstallment'] * data['MonthsInRepayment']
    data['Prepayments'] = data['Prepayment'] // data['MonthlyInstallment']
    data['ScheduledPayments'] = data['MonthlyInstallment'] * (data['MonthsInRepayment'] - data['Prepayments'] + data['MonthsDelinquent'])
    data['PPR'] = (data['ScheduledPayments'] - data['ActualPayments']) / data['CurrentUPB']
    return data
