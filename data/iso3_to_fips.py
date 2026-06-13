"""
Static ISO 3166-1 alpha-3 → FIPS 10-4 country code lookup.

Covers all 238 ISO-3 codes observed in POLECAT (2018-2024).
Returns None for territories with no FIPS equivalent.
"""

ISO3_TO_FIPS: dict[str, str | None] = {
    "ABW": "AA",   # Aruba
    "AFG": "AF",   # Afghanistan
    "AGO": "AO",   # Angola
    "AIA": "AV",   # Anguilla
    "ALA": None,   # Åland Islands (no FIPS)
    "ALB": "AL",   # Albania
    "AND": "AN",   # Andorra
    "ARE": "AE",   # United Arab Emirates
    "ARG": "AR",   # Argentina
    "ARM": "AM",   # Armenia
    "ASM": "AQ",   # American Samoa
    "ATA": "AY",   # Antarctica
    "ATG": "AC",   # Antigua and Barbuda
    "AUS": "AS",   # Australia
    "AUT": "AU",   # Austria
    "AZE": "AJ",   # Azerbaijan
    "BDI": "BY",   # Burundi
    "BEL": "BE",   # Belgium
    "BEN": "BN",   # Benin
    "BES": None,   # Bonaire/Sint Eustatius/Saba (no FIPS)
    "BFA": "UV",   # Burkina Faso
    "BGD": "BG",   # Bangladesh
    "BGR": "BU",   # Bulgaria
    "BHR": "BA",   # Bahrain
    "BHS": "BF",   # Bahamas
    "BIH": "BK",   # Bosnia-Herzegovina
    "BLM": None,   # Saint Barthélemy (no FIPS)
    "BLR": "BO",   # Belarus
    "BLZ": "BH",   # Belize
    "BMU": "BD",   # Bermuda
    "BOL": "BL",   # Bolivia
    "BRA": "BR",   # Brazil
    "BRB": "BB",   # Barbados
    "BRN": "BX",   # Brunei
    "BTN": "BT",   # Bhutan
    "BWA": "BC",   # Botswana
    "CAF": "CF",   # Central African Republic
    "CAN": "CA",   # Canada
    "CCK": "CK",   # Cocos (Keeling) Islands
    "CHE": "SZ",   # Switzerland
    "CHL": "CI",   # Chile
    "CHN": "CH",   # China
    "CIV": "IV",   # Côte d'Ivoire
    "CMR": "CM",   # Cameroon
    "COD": None,   # Congo (DRC) — no current FIPS 10-4 (formerly ZR/Zaire)
    "COG": "CG",   # Congo (Republic)
    "COK": "CW",   # Cook Islands
    "COL": "CO",   # Colombia
    "COM": "CN",   # Comoros
    "CPV": "CV",   # Cape Verde
    "CRI": "CS",   # Costa Rica
    "CUB": "CU",   # Cuba
    "CUW": None,   # Curaçao (no FIPS — formerly Netherlands Antilles)
    "CXR": None,   # Christmas Island (no FIPS)
    "CYM": "CJ",   # Cayman Islands
    "CYP": "CY",   # Cyprus
    "CZE": "EZ",   # Czech Republic
    "DEU": "GM",   # Germany
    "DJI": "DJ",   # Djibouti
    "DMA": "DO",   # Dominica
    "DNK": "DA",   # Denmark
    "DOM": "DR",   # Dominican Republic
    "DZA": "AG",   # Algeria
    "ECU": "EC",   # Ecuador
    "EGY": "EG",   # Egypt
    "ERI": "ER",   # Eritrea
    "ESH": "WI",   # Western Sahara
    "ESP": "SP",   # Spain
    "EST": "EN",   # Estonia
    "ETH": "ET",   # Ethiopia
    "FIN": "FI",   # Finland
    "FJI": "FJ",   # Fiji
    "FLK": "FK",   # Falkland Islands
    "FRA": "FR",   # France
    "FRO": "FO",   # Faroe Islands
    "FSM": "FM",   # Micronesia
    "GAB": "GB",   # Gabon
    "GBR": "UK",   # United Kingdom
    "GEO": "GG",   # Georgia
    "GGY": "GK",   # Guernsey
    "GHA": "GH",   # Ghana
    "GIB": "GI",   # Gibraltar
    "GIN": "GV",   # Guinea
    "GLP": "GP",   # Guadeloupe
    "GMB": "GA",   # Gambia
    "GNB": "PU",   # Guinea-Bissau
    "GNQ": "EK",   # Equatorial Guinea
    "GRC": "GR",   # Greece
    "GRD": "GJ",   # Grenada
    "GRL": "GL",   # Greenland
    "GTM": "GT",   # Guatemala
    "GUF": None,   # French Guiana (no FIPS)
    "GUM": "GQ",   # Guam
    "GUY": "GY",   # Guyana
    "HKG": "HK",   # Hong Kong
    "HND": "HO",   # Honduras
    "HRV": "HR",   # Croatia
    "HTI": "HA",   # Haiti
    "HUN": "HU",   # Hungary
    "IDN": "ID",   # Indonesia
    "IMN": "IM",   # Isle of Man
    "IND": "IN",   # India
    "IOT": None,   # British Indian Ocean Territory (no FIPS)
    "IRL": "EI",   # Ireland
    "IRN": "IR",   # Iran
    "IRQ": "IZ",   # Iraq
    "ISL": "IC",   # Iceland
    "ISR": "IS",   # Israel
    "ITA": "IT",   # Italy
    "JAM": "JM",   # Jamaica
    "JEY": "JE",   # Jersey
    "JOR": "JO",   # Jordan
    "JPN": "JA",   # Japan
    "KAZ": "KZ",   # Kazakhstan
    "KEN": "KE",   # Kenya
    "KGZ": "KG",   # Kyrgyzstan
    "KHM": "CB",   # Cambodia
    "KIR": "KR",   # Kiribati
    "KNA": "SC",   # Saint Kitts and Nevis
    "KOR": "KS",   # South Korea
    "KWT": "KU",   # Kuwait
    "LAO": "LA",   # Laos
    "LBN": "LE",   # Lebanon
    "LBR": "LI",   # Liberia
    "LBY": "LY",   # Libya
    "LCA": "ST",   # Saint Lucia
    "LIE": "LS",   # Liechtenstein
    "LKA": "CE",   # Sri Lanka
    "LSO": "LT",   # Lesotho
    "LTU": "LH",   # Lithuania
    "LUX": "LU",   # Luxembourg
    "LVA": "LG",   # Latvia
    "MAC": "MC",   # Macau
    "MAR": "MO",   # Morocco
    "MCO": "MN",   # Monaco
    "MDA": "MD",   # Moldova
    "MDG": "MA",   # Madagascar
    "MDV": "MV",   # Maldives
    "MEX": "MX",   # Mexico
    "MHL": "RM",   # Marshall Islands
    "MKD": "MK",   # North Macedonia
    "MLI": "ML",   # Mali
    "MLT": "MT",   # Malta
    "MMR": "BM",   # Myanmar
    "MNE": "MJ",   # Montenegro
    "MNG": "MG",   # Mongolia
    "MNP": "CQ",   # Northern Mariana Islands
    "MOZ": "MZ",   # Mozambique
    "MRT": "MR",   # Mauritania
    "MSR": "MH",   # Montserrat
    "MTQ": "MB",   # Martinique
    "MUS": "MP",   # Mauritius
    "MWI": "MI",   # Malawi
    "MYS": "MY",   # Malaysia
    "MYT": "MF",   # Mayotte
    "NA":  None,   # Not a valid ISO3 code
    "NAM": "WA",   # Namibia
    "NCL": "NC",   # New Caledonia
    "NER": "NG",   # Niger
    "NGA": "NI",   # Nigeria
    "NIC": "NU",   # Nicaragua
    "NIU": None,   # Niue (no FIPS)
    "NLD": "NL",   # Netherlands
    "NOR": "NO",   # Norway
    "NPL": "NP",   # Nepal
    "NRU": "NR",   # Nauru
    "NZL": "NZ",   # New Zealand
    "None": None,  # Literal "None" string in POLECAT data
    "OMN": "MU",   # Oman
    "PAK": "PK",   # Pakistan
    "PAN": "PM",   # Panama
    "PCN": None,   # Pitcairn Islands (no FIPS)
    "PER": "PE",   # Peru
    "PHL": "RP",   # Philippines
    "PLW": "PS",   # Palau
    "PNG": "PP",   # Papua New Guinea
    "POL": "PL",   # Poland
    "PRI": "RQ",   # Puerto Rico
    "PRK": "KN",   # North Korea
    "PRT": "PO",   # Portugal
    "PRY": "PA",   # Paraguay
    "PSE": "GZ",   # Palestinian Territories (Gaza/West Bank → GZ as proxy)
    "PYF": "FP",   # French Polynesia
    "QAT": "QA",   # Qatar
    "REU": "RE",   # Réunion
    "ROU": "RO",   # Romania
    "RUS": "RS",   # Russia
    "RWA": "RW",   # Rwanda
    "SAU": "SA",   # Saudi Arabia
    "SCG": "RI",   # Serbia and Montenegro (legacy → Serbia)
    "SDN": "SU",   # Sudan
    "SEN": "SG",   # Senegal
    "SGP": "SN",   # Singapore
    "SHN": "SH",   # Saint Helena
    "SJM": None,   # Svalbard and Jan Mayen (no FIPS)
    "SLB": "BP",   # Solomon Islands
    "SLE": "SL",   # Sierra Leone
    "SLV": "ES",   # El Salvador
    "SMR": "SM",   # San Marino
    "SOM": "SO",   # Somalia
    "SRB": "RI",   # Serbia
    "SSD": None,   # South Sudan (independent 2011, no FIPS 10-4 assigned)
    "STP": "TP",   # São Tomé and Príncipe
    "SUR": "NS",   # Suriname
    "SVK": "LO",   # Slovakia
    "SVN": "SI",   # Slovenia
    "SWE": "SW",   # Sweden
    "SWZ": "WZ",   # Eswatini
    "SYC": "SE",   # Seychelles
    "SYR": "SY",   # Syria
    "TCA": "TK",   # Turks and Caicos Islands
    "TCD": "CD",   # Chad
    "TGO": "TO",   # Togo
    "THA": "TH",   # Thailand
    "TJK": "TI",   # Tajikistan
    "TKM": "TX",   # Turkmenistan
    "TLS": "TT",   # Timor-Leste
    "TON": "TN",   # Tonga
    "TTO": "TD",   # Trinidad and Tobago
    "TUN": "TS",   # Tunisia
    "TUR": "TU",   # Turkey
    "TUV": "TV",   # Tuvalu
    "TWN": "TW",   # Taiwan
    "TZA": "TZ",   # Tanzania
    "UGA": "UG",   # Uganda
    "UKR": "UP",   # Ukraine
    "UMI": None,   # U.S. Minor Outlying Islands (no FIPS)
    "URY": "UY",   # Uruguay
    "USA": "US",   # United States
    "UZB": "UZ",   # Uzbekistan
    "VAT": "VT",   # Vatican City
    "VCT": "VC",   # Saint Vincent and the Grenadines
    "VEN": "VE",   # Venezuela
    "VGB": None,   # British Virgin Islands (no FIPS)
    "VIR": "VI",   # U.S. Virgin Islands
    "VNM": "VM",   # Vietnam
    "VUT": "NH",   # Vanuatu
    "WLF": "WF",   # Wallis and Futuna
    "WSM": "WS",   # Samoa
    "XKX": "KV",   # Kosovo
    "YEM": "YM",   # Yemen
    "ZAF": "SF",   # South Africa
    "ZMB": "ZA",   # Zambia
    "ZWE": "ZI",   # Zimbabwe
}


def iso3_to_fips(iso3: str) -> str | None:
    """Return FIPS 10-4 code for an ISO 3166-1 alpha-3 code, or None if unmapped."""
    return ISO3_TO_FIPS.get(iso3)
