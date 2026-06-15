DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS properties;

CREATE TABLE properties (
    property_id SERIAL PRIMARY KEY,
    address TEXT NOT NULL,
    metro_area TEXT NOT NULL,
    sq_footage INTEGER NOT NULL,
    property_type TEXT NOT NULL
);

CREATE TABLE financials (
    financial_id SERIAL PRIMARY KEY,
    property_id INTEGER REFERENCES properties(property_id),
    revenue NUMERIC,
    net_income NUMERIC,
    expenses NUMERIC
);