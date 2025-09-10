import pandas as pd
from difflib import get_close_matches
import json
from pathlib import Path
import networkx as nx
import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# Station Data
# =========================
lrt_kelana_jaya_line = [
    'Gombak','Taman Melati','Wangsa Maju','Sri Rampai','Setiawangsa','Jelatek',
    "Dato' Keramat",'Damai','Ampang Park','KLCC','Kampung Baru','Dang Wangi',
    'Masjid Jamek','Pasar Seni','KL Sentral', 'Bangsar', 'Abdullah Hukum', 'Kerinchi',
    'Universiti','Taman Jaya','Asia Jaya','Taman Paramount','Taman Bahagia','Kelana Jaya',
    'Lembah Subang','Ara Damansara', 'Glenmarie','Subang Jaya','SS 15','SS 18',
    'USJ 7','Taipan','Wawasan','USJ 21','Alam Megah','Subang Alam','Putra Heights'
]

brt_sunway_line = [
    "Setia Jaya", "Mentari", "Sunway Lagoon", "SunMed", "SunU Monash", "South Quay USJ1", "USJ 7"
]

lrt_ampang_line = [
    "Sentul Timur", "Sentul", "Titiwangsa", "PWTC", "Sultan Ismail", "Bandaraya", "Masjid Jamek", "Plaza Rakyat",
    "Hang Tuah", "Pudu", "Chan Sow Lin", "Miharja", "Maluri", "Pandan Jaya", "Pandan Indah", "Cempaka",
    "Cahaya", "Ampang"
]

# IMPORTANT: no trailing comma here (it would turn it into a tuple)
kl_monorail = [
    "KL Sentral Monorail", "Tun Sambanthan", "Maharajalela", "Hang Tuah", "Imbi", "Bukit Bintang",
    "Raja Chulan", "Bukit Nanas", "Medan Tuanku", "Chow Kit", "Titiwangsa"
]

lrt_sri_petaling_line = [
    "Sentul Timur", "Sentul", "Titiwangsa", "PWTC", "Sultan Ismail", "Bandaraya", "Masjid Jamek", "Plaza Rakyat",
    "Hang Tuah", "Pudu", "Chan Sow Lin", "Cheras", "Salak Selatan", "Bandar Tun Razak", "Tasik Selatan",
    "Sungai Besi", "Bukit Jalil", "Sri Petaling", "Awan Besar", "Muhibbah", "Alam Sutera", "Kinrara BK5",
    "IOI Puchong Jaya", "Pusat Bandar Puchong", "Taman Perindustrian Puchong", "Bandar Puteri",
    "Puchong Perdana", "Puchong Prima", "Putra Heights"
]

mrt_kajang_line = [
    'Sungai Buloh','Kampung Selamat','Kwasa Damansara','Kwasa Sentral','Kota Damansara',
    'Surian','Mutiara Damansara','Bandar Utama','TTDI','Phileo Damansara','Pusat Bandar Damansara',
    'Semantan','Muzium Negara','Pasar Seni','Merdeka','Bukit Bintang','Tun Razak Exchange',
    'Cochrane','Maluri','Taman Pertama','Taman Midah','Taman Mutiara','Taman Connaught',
    'Taman Suntex','Sri Raya','Bandar Tun Hussein Onn','Batu 11 Cheras','Bukit Dukung',
    'Sungai Jernih','Stadium Kajang','Kajang'
]

mrt_putrajaya_line = [
    "Kwasa Damansara", "Kampung Selamat", "Sungai Buloh", "Damansara Damai", "Sri Damansara Barat", 
    "Sri Damansara Sentral", "Sri Damansara Timur", "Metro Prima", "Kepong Baru", "Jinjang",
    "Sri Delima", "Kampung Batu", "Kentonmen", "Jalan Ipoh", "Sentul Barat", "Titiwangsa",
    "Hospital Kuala Lumpur", "Raja Uda", "Ampang Park", "Persiaran KLCC", "Conlay", "Tun Razak Exchange",
    "Chan Sow Lin", "Kuchai", "Taman Naga Emas", "Sungai Besi", "Serdang Raya Utara", "Serdang Raya Sentral",
    "Serdang Raya Selatan", "Serdang Jaya", "UPM", "Taman Equine", "Putra Permai", "16 Sierra",
    "Cyberjaya Utara", "Cyber City Centre", "Putrajaya Sentral"
]

interchanges = {
  "Ampang Park": ["LRT Kelana Jaya Line", "MRT Putrajaya Line"],
  "Masjid Jamek": ["LRT Kelana Jaya Line", "LRT Ampang Line", "LRT Sri Petaling Line"],
  "Hang Tuah": ["LRT Ampang Line", "LRT Sri Petaling Line", "KL Monorail"],
  "Chan Sow Lin": ["LRT Ampang Line", "LRT Sri Petaling Line", "MRT Putrajaya Line"],
  "Pasar Seni": ["LRT Kelana Jaya Line", "MRT Kajang Line", "LRT Sri Petaling Line"],
  "Bukit Bintang": ["KL Monorail", "MRT Kajang Line"],
  "Muzium Negara": ["KL Monorail", "MRT Kajang Line"],
  "Sungai Besi": ["LRT Sri Petaling Line", "MRT Putrajaya Line"],
  "Sungai Buloh": ["MRT Kajang Line", "MRT Putrajaya Line"],
  "Putra Heights": ["LRT Sri Petaling Line", "LRT Kelana Jaya Line"],
  "USJ 7": ["BRT Sunway Line", "LRT Kelana Jaya Line"],
  "Kwasa Damansara": ["MRT Putrajaya Line", "MRT Kajang Line"],
  "Titiwangsa":["KL Monorail", "MRT Putrajaya Line", "LRT Sri Petaling Line", "LRT Ampang Line"],
  "Tun Razak Exchange": ["MRT Putrajaya Line", "MRT Kajang Line"]
}

# =========================
# Time Travelled setup
# =========================

# Minutes after midnight helper
def _mins(h, m): return h*60 + m

TRAIN_HEADWAYS = {
    "weekday": [
        (_mins(6,0),  _mins(7,0),   8),
        (_mins(7,0),  _mins(9,30),  5),
        (_mins(9,30), _mins(17,0), 10),
        (_mins(17,0), _mins(19,30), 5),
        (_mins(19,30),_mins(24,0), 10),  # 24:00 = end of day
    ],
    "saturday": [
        (_mins(6,0),  _mins(24,0), 10),
    ],
    "sundayph": [
        (_mins(6,0),  _mins(23,30), 10),
    ],
}

def headway_minutes(now: datetime.datetime, *, is_public_holiday: bool = False) -> int:
    """Return scheduled headway (minutes) for the given datetime."""
    weekday = now.weekday()  # 0=Mon .. 6=Sun
    period = "weekday" if weekday < 5 else ("saturday" if weekday == 5 else "sundayph")
    if is_public_holiday:
        period = "sundayph"

    minute_of_day = now.hour*60 + now.minute
    for start, end, hw in TRAIN_HEADWAYS[period]:
        # interval is [start, end)
        if start <= minute_of_day < end:
            return hw
    # If outside service windows, fall back to the last period's headway
    return TRAIN_HEADWAYS[period][-1][2]

# =========================
# FAISS setup for station search
# =========================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

lines_data = {
    "LRT Kelana Jaya Line": lrt_kelana_jaya_line,
    "MRT Kajang Line": mrt_kajang_line,
    "BRT Sunway Line": brt_sunway_line,
    "LRT Ampang Line": lrt_ampang_line,
    "LRT Sri Petaling Line": lrt_sri_petaling_line,
    "MRT Putrajaya Line": mrt_putrajaya_line,
    "KL Monorail": kl_monorail
}

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

station_index = build_station_index(lines_data)

# =========================
# Station Finder
# =========================
def find_station(user_station: str):
    """
    Improved station finder:
    1) Exact match (case-insensitive)
    2) Whole-word prefix match (shorter names win)
    3) Fuzzy matching (difflib)
    4) Semantic FAISS fallback
    Returns: (station_name, line_name, is_interchange: bool)
    """
    if not user_station:
        return None, None, None

    user_station_lower = user_station.strip().lower()

    # Flatten stations
    all_stations = [(station, line) for line, stations in lines_data.items() for station in stations]
    station_names = [station for station, _ in all_stations]

    # 1) Exact
    for line_name, stations in lines_data.items():
        for station in stations:
            if station.lower() == user_station_lower:
                return station, line_name, station in interchanges

    # 2) Whole-word prefix (avoid "Stadium Kajang" for "Kajang")
    candidates = []
    for line_name, stations in lines_data.items():
        for station in stations:
            if station.lower().split()[0] == user_station_lower:
                candidates.append((station, line_name))
    if candidates:
        station, line_name = sorted(candidates, key=lambda x: len(x[0]))[0]
        return station, line_name, station in interchanges

    # 3) Fuzzy
    close_matches = get_close_matches(user_station_lower, [s.lower() for s in station_names], n=1, cutoff=0.6)
    if close_matches:
        matched_name = close_matches[0]
        for station, line in all_stations:
            if station.lower() == matched_name:
                return station, line, station in interchanges

    # 4) FAISS
    results = station_index.similarity_search_with_score(user_station, k=1)
    if results:
        best_match, score = results[0]
        SIMILARITY_THRESHOLD = 0.75
        if score >= SIMILARITY_THRESHOLD:
            return (
                best_match.metadata["station"],
                best_match.metadata["line"],
                best_match.metadata["interchange"]
            )

    return None, None, None

# =========================
# Transit Graph
# =========================
def build_transit_graph(per_stop: float = 0.40, interchange_cost: float = 0.0) -> nx.Graph:
    """
    Build a multi-line graph:
      - Nodes are disambiguated as "Station (Line)"
      - Adjacent stations on the same line have weight = per_stop
      - Same-named station across multiple lines are connected with interchange_cost
    """
    G = nx.Graph()

    # Same-line adjacencies
    for line, stations in lines_data.items():
        for i in range(len(stations) - 1):
            a = f"{stations[i]} ({line})"
            b = f"{stations[i+1]} ({line})"
            G.add_edge(a, b, weight=per_stop)

    # Interchange links
    for station, lines in interchanges.items():
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a = f"{station} ({lines[i]})"
                b = f"{station} ({lines[j]})"
                G.add_edge(a, b, weight=interchange_cost)

    return G

# =========================
# Fare System
# =========================
class FareSystem:
    """
    Uses CSV fares when available. Otherwise, estimates fare via shortest path over the transit graph.
    """
    def __init__(self, csv_path, from_station):
        self.df = pd.read_csv(csv_path, index_col=0)
        
        # Normalize station names
        self.df.index = self.df.index.astype(str).str.strip()
        self.df.columns = self.df.columns.astype(str).str.strip()

        self.graph = build_transit_graph()

        station, line, _ = find_station(from_station)
        if not station or not line:
            raise ValueError(f"Invalid from_station: '{from_station}' not recognized.")
        self.from_station_name = station                # plain name (for CSV)
        self.from_node = f"{station} ({line})"          # disambiguated for graph

    def get_fare(self, to_station):
        """
        Try exact CSV fare using plain station names.
        If missing, estimate via graph shortest path between disambiguated nodes.
        Returns: (fare: float, to_line: str, is_estimated: bool)
        """
        station, line, _ = find_station(to_station)
        if not station or not line:
            return None, None, None

        # Try CSV (only if both stations exist in table)
        if self.from_station_name in self.df.index and station in self.df.columns:
            try:
                fare = self.df.loc[self.from_station_name, station]
                if pd.notna(fare):
                    return float(fare), line, False
            except Exception:
                pass  # fall through to estimation

        # Estimate
        est_fare, _ = self.estimate_fare(self.from_node, f"{station} ({line})")
        return est_fare, line, True
    
    def estimate_fare(
        self,
        from_node: str,
        to_node: str,
        base_fare: float = 1.00
    ) -> tuple[float, list]:
        """
        Estimate fare.

        Formula:
            fare = base_fare + total_weight
            (total_weight accumulates per-stop costs & interchange costs from the graph)

        Returns:
            tuple: (fare, path_nodes)
        """
        try:
            path = nx.shortest_path(
                self.graph, from_node, to_node, weight="weight"
            )
            total_weight = nx.path_weight(
                self.graph, path, weight="weight"
            )
            return round(base_fare + total_weight, 2), path
        except nx.NetworkXNoPath:
            return 5.00, []


    def estimate_travel_time(
        self,
        to_station: str,
        when: datetime.datetime,
        *,
        interstation_min: float = 2.7,      # avg minutes per inter-station hop (incl. dwell)
        interchange_walk_min: float = 2.0,  # avg platform-to-platform walk per interchange
        count_wait_at_interchanges: bool = True,
        is_public_holiday: bool = False,
    ) -> tuple[float, dict]:
        """
        Returns (total_minutes, breakdown_dict).
        Time model = hops * interstation_min
                     + (half headway) per boarding (origin + each interchange if enabled)
                     + interchange_walk_min per interchange
        """
        # Resolve destination
        station, line, _ = find_station(to_station)
        if not station or not line:
            return None, {"error": f"Unknown destination: {to_station}"}

        # Path on the existing fare graph (nodes are 'Name (Line)')
        try:
            path = nx.shortest_path(self.graph, self.from_node, f"{station} ({line})", weight="weight")
        except nx.NetworkXNoPath:
            return None, {"error": "No route found"}

        # Count hops (edges)
        hops = max(0, len(path) - 1)

        # Extract lines along the path to count boardings/interchanges
        step_lines = [p.split(" (")[1].rstrip(")") for p in path]
        interchanges_count = sum(
            1 for i in range(1, len(step_lines)) if step_lines[i] != step_lines[i-1]
        )
        boardings = 1 + interchanges_count  # origin + each line change

        # In-train travel
        in_train = hops * interstation_min

        # Waiting time (same headway for all lines per your table)
        hw = headway_minutes(when, is_public_holiday=is_public_holiday)
        waits = (boardings if count_wait_at_interchanges else 1) * (hw / 2.0)

        # Interchange walking time
        walks = interchanges_count * interchange_walk_min

        total = round(in_train + waits + walks, 1)
        breakdown = {
            "stops": hops,
            "boardings": boardings,
            "interchanges": interchanges_count,
            "headway_min": hw,
            "in_train_min": round(in_train, 1),
            "wait_min": round(waits, 1),
            "walk_min": round(walks, 1),
            "path": path,
        }
        return total, breakdown

    def get_all_stations(self):
        # Only those present in the CSV (UI helper)
        return [s for s in self.df.columns.tolist() if s != self.from_station_name]

    def print_fares_to_all_destinations(self):
        print(f"Fares from {self.from_station_name}:")
        for dest in self.get_all_stations():
            fare = self.df.loc[self.from_station_name, dest]
            print(f"  To {dest}: RM{fare:.2f}" if fare is not None else f"  To {dest}: Fare unavailable")

# =========================
# RAG Knowledge Base for QnA
# =========================
class KnowledgeBase:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.data = self._load_data()
        self.retriever = self._build_retriever()

    def _load_data(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_retriever(self):
        texts, metadatas = [], []
        for idx, entry in enumerate(self.data):
            q = entry["question"]
            a = entry["answer"]
            topic = entry.get("topic", "General")

            texts.append(q)  # use question as retrieval text
            metadatas.append({
                "id": idx,
                "topic": topic,
                "answer": a,
                "question": q
            })

        return FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

    def query(self, user_input: str, k: int = 2):
        """Return top-k relevant QA pairs."""
        results = self.retriever.similarity_search(user_input, k=k)
        return [
            {
                "question": r.metadata["question"],
                "answer": r.metadata["answer"],
                "topic": r.metadata["topic"]
            }
            for r in results
        ]


# Initialize globally (so it loads once)
rules_facilities_kb = KnowledgeBase("lrt_mrt_general_rules_facilities.json")


# =========================
# Route Planning (multi-line, multi-interchange)
# =========================
def plan_route(from_station: str, to_station: str, fare_system: FareSystem):
    """
    Returns:
      route_text_en: str
      route_text_ms: str
      station_lines: List[List[str]]   # segments per line in order
      interchanges: List[str]          # ordered list of interchange station names
    """
    start, start_line, _ = find_station(from_station)
    end, end_line, _ = find_station(to_station)

    if not start or not end:
        return (
            "One or both stations could not be found.",
            "Satu atau kedua-dua stesen tidak dapat dijumpai.",
            [],
            []
        )

    # Shortest path on graph between disambiguated nodes
    est_fare, path = fare_system.estimate_fare(
        f"{start} ({start_line})",
        f"{end} ({end_line})"
    )

    if not path:
        return (
            f"No route found between {start} and {end}.",
            f"Tiada laluan dijumpai antara {start} dan {end}.",
            [],
            []
        )
        
    # Estimate travel time & stops
    dt = datetime.datetime.now()
    time, breakdown = fare_system.estimate_travel_time(end, dt)
    stops = breakdown.get("stops", 0)

    # Try official fare first
    fare_val, _, is_est = fare_system.get_fare(end)
    if fare_val is not None:
        fare_to_use = fare_val
    else:
        fare_to_use = est_fare  # fallback

    # Split path into station names and their lines
    step_stations = [p.split(" (")[0] for p in path]
    step_lines = [p.split(" (")[1].strip(")") for p in path]

    # Build contiguous segments per line
    station_lines: list[list[str]] = []
    current_line = step_lines[0]
    current_segment: list[str] = []

    for st, ln in zip(step_stations, step_lines):
        if ln == current_line:
            current_segment.append(st)
        else:
            if current_segment:
                station_lines.append(current_segment)
            current_line = ln
            current_segment = [st]
    if current_segment:
        station_lines.append(current_segment)

    # --- FIX: merge stray 1-station segments forward ---
    merged_segments = []
    for seg in station_lines:
        if merged_segments and len(seg) == 1:
            merged_segments[-1].extend(seg)
        else:
            merged_segments.append(seg)
    station_lines = merged_segments

    # Collect interchanges
    interchanges_list: list[str] = []
    for i in range(1, len(step_stations)):
        if step_lines[i] != step_lines[i - 1]:
            if step_stations[i] in interchanges:
                interchanges_list.append(step_stations[i])

    # Deduplicate while preserving order
    seen = set()
    ordered_inters = []
    for s in interchanges_list:
        if s not in seen:
            ordered_inters.append(s)
            seen.add(s)

    # Build human text
    if ordered_inters:
        inter_en = " via " + ", ".join(ordered_inters)
        inter_ms = " melalui " + ", ".join(ordered_inters)
    else:
        inter_en = ""
        inter_ms = ""
        
    route_text_en = (
        f"Here is the route from {start} to {end}{inter_en}."
        f"\nThe estimated travel time is {time} minutes."
        f"\nThere will be {stops} stops."
        f"\nThe total fare is RM{fare_to_use:.2f}"
    )
    route_text_ms = (
        f"Berikut ialah laluan dari {start} ke {end}{inter_ms}."
        f"\nAnggaran masa perjalanan ialah {time} minit."
        f"\nAkan ada {stops} hentian."
        f"\nJumlah tambang ialah RM{fare_to_use:.2f}"
    )

    return route_text_en, route_text_ms, station_lines, ordered_inters

