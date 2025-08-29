import pandas as pd
from difflib import get_close_matches

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Station data
lrt_kelana_jaya_line = [
    'Gombak','Taman Melati','Wangsa Maju','Sri Rampai','Setiawangsa','Jelatek',
    "Dato' Keramat",'Damai','Ampang Park','KLCC','Kampung Baru','Dang Wangi',
    'Masjid Jamek','Pasar Seni','KL Sentral', 'Bangsar', 'Abdullah Hukum', 'Kerinchi',
    'Universiti','Taman Jaya','Asia Jaya','Taman Paramount','Taman Bahagia','Kelana Jaya',
    'Lembah Subang','Ara Damansara', 'Glenmarie','Subang Jaya','SS 15','SS 18',
    'USJ 7','Taipan','Wawasan','USJ 21','Alam Megah','Subang Alam','Putra Heights'
]

mrt_kajang_line = [
    'Sungai Buloh','Kampung Selamat','Kwasa Damansara','Kwasa Sentral','Kota Damansara',
    'Surian','Mutiara Damansara','Bandar Utama','TTDI','Phileo Damansara','Pusat Bandar Damansara',
    'Semantan','Muzium Negara','Pasar Seni','Merdeka','Bukit Bintang','Tun Razak Exchange (TRX)',
    'Cochrane','Maluri','Taman Pertama','Taman Midah','Taman Mutiara','Taman Connaught',
    'Taman Suntex','Sri Raya','Bandar Tun Hussein Onn','Batu 11 Cheras','Bukit Dukung',
    'Sungai Jernih','Stadium Kajang','Kajang'
]

interchanges = {
    "Pasar Seni": ["LRT Kelana Jaya Line", "MRT Kajang Line"]
}

# FAISS setup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_station_index(line_data):
    texts, metadatas = [], []
    for line_name, stations in line_data.items():
        for station in stations:
            texts.append(f"{station} ({line_name})")
            metadatas.append({
                "station": station,
                "line": line_name,
                "interchange": station in interchanges
            })
    return FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

lines_data = {
    "LRT Kelana Jaya Line": lrt_kelana_jaya_line,
    "MRT Kajang Line": mrt_kajang_line,
}
station_index = build_station_index(lines_data)

def find_station(user_station: str):
    """
    Improved station finder:
    1. Try exact case-insensitive match.
    2. Try whole-word prefix match (prefer shorter station names first).
    3. Try fuzzy matching for misspellings.
    4. Fall back to FAISS semantic search.
    """
    user_station_lower = user_station.strip().lower()

    # Get all station names for fuzzy matching
    all_stations = [(station, line) for line, stations in lines_data.items() for station in stations]
    station_names = [station for station, _ in all_stations]

    # Step 1: Exact match (case-insensitive)
    for line_name, stations in lines_data.items():
        for station in stations:
            if station.lower() == user_station_lower:
                return station, line_name, station in interchanges

    # Step 2: Whole-word prefix match (avoid "Stadium Kajang" when user types "Kajang")
    candidates = []
    for line_name, stations in lines_data.items():
        for station in stations:
            station_lower = station.lower()
            # Split into words so "Kajang" doesn't match "Stadium Kajang"
            if station_lower.split()[0] == user_station_lower:
                candidates.append((station, line_name))
    if candidates:
        # Sort by length → "Kajang" preferred over "Stadium Kajang"
        station, line_name = sorted(candidates, key=lambda x: len(x[0]))[0]
        return station, line_name, station in interchanges

    # Step 3: Fuzzy matching for misspellings
    close_matches = get_close_matches(user_station_lower, [s.lower() for s in station_names], n=1, cutoff=0.6)
    if close_matches:
        matched_name = close_matches[0]
        for station, line in all_stations:
            if station.lower() == matched_name:
                return station, line, station in interchanges

    # Step 4: Fall back to FAISS semantic search
    results = station_index.similarity_search_with_score(user_station, k=1)
    if results:
        best_match, score = results[0]
        return (
            best_match.metadata["station"],
            best_match.metadata["line"],
            best_match.metadata["interchange"]
        )

    return None, None, None

class FareSystem:
    def __init__(self, csv_path, from_station):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.stations = self.df.columns.tolist()

        # Fuzzy match kiosk station
        station, _, _ = find_station(from_station)
        if station in self.stations:
            self.from_station = station
        else:
            raise ValueError(f"Invalid from_station: '{from_station}' not found in fare table.")
    
    def get_fare(self, to_station):
        # Fuzzy match destination
        station, line, _ = find_station(to_station)
        if not station:
            return None, None, None
        try:
            fare = self.df.loc[self.from_station, station]
            return fare, line, False
        except KeyError:
            return None, line, False
    
    def get_all_stations(self):
        return [station for station in self.stations if station != self.from_station]
    
    def print_fares_to_all_destinations(self):
        print(f"Fares from {self.from_station}:")
        for dest in self.get_all_stations():
            fare = self.df.loc[self.from_station, dest]
            print(f"  To {dest}: RM{fare:.2f}" if fare is not None else f"  To {dest}: Fare unavailable")
    
    def find_interchange_station(self, to_station):
        """Return the interchange station name if travel involves switching lines."""
        from_station_name, from_line, _ = find_station(self.from_station)
        to_station_name, to_line, _ = find_station(to_station)

        # If on same line, no interchange needed
        if from_line == to_line:
            return None
        
        # Look for an interchange station connecting both lines
        for station, lines in interchanges.items():
            if from_line in lines and to_line in lines:
                return station
        
        return None

# Route Planning
def plan_route(from_station: str, to_station: str, fare_system: FareSystem) -> str:
    """
    Plan a route between two stations.
    Shows step-by-step route with line changes, fare, and interchanges.
    """
    start, start_line, _ = find_station(from_station)
    end, end_line, _ = find_station(to_station)

    if not start or not end:
        return "One or both stations could not be found.", "Satu atau kedua-dua stesen tidak dapat dijumpai.", [], [], ""

    # === Case 1: Same line ===
    if start_line == end_line:
        line_stations = lines_data[start_line]
        start_idx = line_stations.index(start)
        end_idx = line_stations.index(end)
        step_stations = line_stations[start_idx:end_idx+1] if start_idx < end_idx else line_stations[end_idx:start_idx+1][::-1]

        steps = [f"🚇 {start_line}"]
        for st in step_stations:
            steps.append(f"➡️ {st}")
        fare, _, _ = fare_system.get_fare(end)
        
        route_text_en = (
            f"Here is the route from {start} to {end}."
            f"\nThe total fare is RM{fare:.2f}"
        )
        route_text_ms = (
            f"Berikut ialah laluan dari {start} ke {end}."
            f"\nJumlah tambang ialah RM{fare:.2f}"
        )
        
        return route_text_en, route_text_ms, step_stations, [], ""

    # === Case 2: Different lines ===
    interchange = fare_system.find_interchange_station(end)
    if not interchange:
        return f"No direct interchange found between {start_line} and {end_line}.", f"Tidak ada pertukaran langsung dijumpai antara {start_line} dan {end_line}.", [], [], ""

    # Step 1: Segment 1 (start → interchange)
    line1_stations = lines_data[start_line]
    s_idx = line1_stations.index(start)
    i_idx = line1_stations.index(interchange)
    seg1 = line1_stations[s_idx:i_idx+1] if s_idx < i_idx else line1_stations[i_idx:s_idx+1][::-1]

    # Step 2: Segment 2 (interchange → end)
    line2_stations = lines_data[end_line]
    i2_idx = line2_stations.index(interchange)
    e_idx = line2_stations.index(end)
    seg2 = line2_stations[i2_idx:e_idx+1] if i2_idx < e_idx else line2_stations[e_idx:i2_idx+1][::-1]

    #steps = []
    #steps.append(f"🚇 {start_line}")
    #for st in seg1:
    #    steps.append(f"➡️ {st}")
    #steps.append(f"🔄 Change at {interchange} to 🚇 {end_line}")
    #for st in seg2[1:]:  # skip duplicate interchange
    #    steps.append(f"➡️ {st}")

    fare, _, _ = fare_system.get_fare(end)

    route_text_en = (
        f"Here is the route from {start} to {end}."
        f"Please note that there will be interchange at {interchange}."
        f"\nThe total fare is RM{fare:.2f}"
    )
    
    route_text_ms = (
        f"Berikut ialah laluan dari {start} ke {end}."
        f"Sila ambil perhatian bahawa akan ada pertukaran di {interchange}."
        f"\nJumlah tambang ialah RM{fare:.2f}"
    )

    return route_text_en, route_text_ms, seg1, seg2, interchange