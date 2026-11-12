-- Default rate by credit band
SELECT credit_band, COUNT(*) AS applications,
       ROUND(AVG(rent_to_income), 3) AS avg_rent_to_income,
       ROUND(AVG(default_12m), 4) AS default_rate
FROM applications GROUP BY credit_band ORDER BY credit_band;

-- Rent burden groups
SELECT CASE WHEN rent_to_income < 0.3 THEN 'under 30%'
            WHEN rent_to_income < 0.5 THEN '30%-49%'
            ELSE '50%+' END AS rent_burden,
       COUNT(*) AS applications, ROUND(AVG(default_12m), 4) AS default_rate
FROM applications GROUP BY rent_burden ORDER BY rent_burden;

-- Employment tenure groups
SELECT CASE WHEN employment_tenure_months < 12 THEN 'under 1 year'
            WHEN employment_tenure_months < 36 THEN '1-3 years'
            ELSE '3+ years' END AS tenure_group,
       COUNT(*) AS applications, ROUND(AVG(default_12m), 4) AS default_rate
FROM applications GROUP BY tenure_group ORDER BY tenure_group;
