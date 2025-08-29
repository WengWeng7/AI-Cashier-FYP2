# file: metro_data.py
metro_lines = {
    "KTM Seremban Line": [
        "Batu Caves", "Taman Wahyu", "Kampung Batu", "Batu Kentonmen", "Sentul", "Bank Negara", "Kuala Lumpur", "KL Sentral",
        "Mid Valley", "Seputeh", "Salak Selatan", "Bandar Tasik Selatan", "Serdang", "Kajang", "UKM", "Bangi",
        "Batang Benar", "Nilai", "Labu", "Tiroi", "Sendayan", "Seremban", "Sungai Gadut", "Rantau", "Senawang",
        "Sungai Gadut", "Gemas"
    ],
    "KTM Port Klang Line": [
        "Tanjung Malim", "Kuala Kubu Bharu", "Rasa", "Batang Kali", "Serendah", "Rawang", "Kuang", "Kundang",
        "Sungai Buloh", "Kepong Sentral", "Kepong", "Segambut", "Putra", "Bank Negara", "Kuala Lumpur", "KL Sentral",
        "Abdullah Hukum", "Pantai Dalam", "Angkasapuri", "Jalan Templer", "Kampung Dato Harun", "Seri Setia",
        "Setia Jaya", "Batu Tiga", "Shah Alam", "Padang Jawa", "Bukit Badak", "Klang", "Teluk Pulai", "Teluk Gadong",
        "Kampung Raja Uda", "Jalan Kastam", "Pelabuhan Klang"
    ],
    "LRT Ampang Line": [
        "Sentul Timur", "Sentul", "Titiwangsa", "PWTC", "Sultan Ismail", "Bandaraya", "Masjid Jamek", "Plaza Rakyat",
        "Hang Tuah", "Pudu", "Chan Sow Lin", "Miharja", "Maluri", "Pandan Jaya", "Pandan Indah", "Cempaka",
        "Cahaya", "Ampang"
    ],
    "LRT Sri Petaling Line": [
        "Sentul Timur", "Sentul", "Titiwangsa", "PWTC", "Sultan Ismail", "Bandaraya", "Masjid Jamek", "Plaza Rakyat",
        "Hang Tuah", "Pudu", "Chan Sow Lin", "Cheras", "Salak Selatan", "Bandar Tun Razak", "Tasik Selatan",
        "Sungai Besi", "Bukit Jalil", "Sri Petaling", "Awan Besar", "Muhibbah", "Alam Sutera", "Kinrara BK5",
        "IOI Puchong Jaya", "Pusat Bandar Puchong", "Taman Perindustrian Puchong", "Bandar Puteri",
        "Puchong Perdana", "Puchong Prima", "Putra Heights"
    ],
    "LRT Kelana Jaya Line": [
        "Gombak", "Taman Melati", "Wangsa Maju", "Sri Rampai", "Setiawangsa", "Jelatek", "Dato’ Keramat", "Damai",
        "Ampang Park", "KLCC", "Kampung Baru", "Dang Wangi", "Masjid Jamek", "Pasar Seni", "KL Sentral", "Bangsar",
        "Abdullah Hukum", "Kerinchi", "Universiti", "Taman Jaya", "Asia Jaya", "Taman Paramount", "Taman Bahagia",
        "Kelana Jaya", "Lembah Subang", "Ara Damansara", "Glenmarie", "Subang Jaya", "SS15", "SS18", "USJ7",
        "Taipan", "Wawasan", "USJ21", "Alam Megah", "Subang Alam", "Putra Heights"
    ],
    "KLIA Ekspres": [
        "KL Sentral", "KLIA", "KLIA2"
    ],
    "KLIA Transit": [
        "KL Sentral", "Bandar Tasik Selatan", "Putrajaya & Cyberjaya", "Salak Tinggi", "KLIA", "KLIA2"
    ],
    "KL Monorail": [
        "KL Sentral Monorail", "Tun Sambanthan", "Maharajalela", "Hang Tuah", "Imbi", "Bukit Bintang",
        "Raja Chulan", "Bukit Nanas", "Medan Tuanku", "Chow Kit", "Titiwangsa"
    ],
    "MRT Kajang Line": [
        "Kwasa Damansara", "Kwasa Sentral", "Kota Damansara", "Surian", "Mutiara Damansara", "Bandar Utama", "TTDI",
        "Phileo Damansara", "Pusat Bandar Damansara", "Semantan", "Muzium Negara", "Pasar Seni", "Merdeka",
        "Bukit Bintang", "Tun Razak Exchange", "Cochrane", "Maluri", "Taman Pertama", "Taman Midah",
        "Taman Mutiara", "Taman Connaught", "Taman Suntex", "Sri Raya", "Bandar Tun Hussein Onn", "Batu 11 Cheras",
        "Bukit Dukung", "Sungai Jernih", "Stadium Kajang", "Kajang"
    ],
    "MRT Putrajaya Line": [
        "Kwasa Damansara", "Kepong Baru", "Titiwangsa", "Chan Sow Lin", "Ampang Park", "Persiaran KLCC",
        "Putrajaya Sentral"
    ],
    "KTM Skypark Link": [
        "KL Sentral", "Subang Jaya", "Terminal Skypark"
    ],
    "BRT Sunway Line": [
        "Setia Jaya", "Mentari", "Sunway Lagoon", "SunMed", "SunU Monash", "South Quay USJ1", "USJ7"
    ]
}

# Updated real physical interchange connections based on Klang Valley Integrated Transit Map
interchange_connections = [
    ("KL Sentral", "Muzium Negara"),
    ("KL Sentral", "Pasar Seni"),
    ("Pasar Seni", "Masjid Jamek"),
    ("Masjid Jamek", "Bandaraya"),
    ("Hang Tuah", "Bukit Bintang"),
    ("Hang Tuah", "Chan Sow Lin"),
    ("Chan Sow Lin", "Tun Razak Exchange"),
    ("Ampang Park", "Persiaran KLCC"),
    ("Titiwangsa", "Sentul"),
    ("Titiwangsa", "PWTC"),
    ("PWTC", "Sultan Ismail"),
    ("Sultan Ismail", "Bandaraya"),
    ("KL Sentral", "Hang Tuah"),
    ("Bandar Tasik Selatan", "Serdang"),
    ("Bandar Tasik Selatan", "Terminal Bersepadu Selatan"),
    ("Maluri", "Cochrane"),
    ("Putrajaya Sentral", "Cyberjaya"),
    ("KLIA", "KLIA2"),
    ("Subang Jaya", "SS15"),
    ("SS15", "SS18"),
    ("SS18", "USJ7"),
    ("USJ7", "Taipan"),
    ("USJ7", "South Quay USJ1"),
    ("South Quay USJ1", "SunU Monash"),
    ("SunU Monash", "SunMed"),
    ("SunMed", "Sunway Lagoon"),
    ("Sunway Lagoon", "Mentari"),
    ("Mentari", "Setia Jaya"),
    ("Setia Jaya", "Seri Setia"),
    ("Seri Setia", "Kampung Dato Harun"),
    ("Kampung Dato Harun", "Jalan Templer"),
    ("Jalan Templer", "Angkasapuri"),
    ("Angkasapuri", "Pantai Dalam"),
    ("Pantai Dalam", "Abdullah Hukum"),
    ("Abdullah Hukum", "Kerinchi"),
    ("Bukit Bintang", "Merdeka"),
    ("Merdeka", "Plaza Rakyat"),
    ("Plaza Rakyat", "Hang Tuah"),
    ("KLCC", "Ampang Park")
]
