import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from battery_models import BatterySpec, PRESET_BATTERIES
from data_services import (
    DEFAULT_ENTSOE_ZONE,
    DEFAULT_FRANK_OPSLAG,
    DEFAULT_PRICE_EXPORT,
    DEFAULT_PRICE_IMPORT,
    load_energy_csv,
    load_price_csv,
    download_entsoe_prices_for_period,
    align_prices_to_energy,
)
from simulation_service import simulate_battery, calculate_financials


class BatterySimulatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thuisbatterij Optimizer")
        self.root.geometry("1480x920")

        self.df: Optional[pd.DataFrame] = None
        self.price_df: Optional[pd.DataFrame] = None
        self.simulations: Dict[str, pd.DataFrame] = {}
        self.metrics_per_battery: Dict[str, Dict[str, float]] = {}
        self.financials_per_battery: Dict[str, Dict[str, float]] = {}
        self.all_batteries: List[BatterySpec] = [BatterySpec(**vars(b)) for b in PRESET_BATTERIES]
        self.summary_df: Optional[pd.DataFrame] = None

        self.import_price_var = tk.DoubleVar(value=DEFAULT_PRICE_IMPORT)
        self.export_price_var = tk.DoubleVar(value=DEFAULT_PRICE_EXPORT)
        self.initial_soc_var = tk.DoubleVar(value=0.0)
        self.frank_opslag_var = tk.DoubleVar(value=DEFAULT_FRANK_OPSLAG)
        self.price_mode_var = tk.StringVar(value="fixed")
        self.entsoe_token_var = tk.StringVar(value="")
        self.entsoe_zone_var = tk.StringVar(value=DEFAULT_ENTSOE_ZONE)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Button(top, text="Verbruiks-CSV kiezen", command=self.load_csv).pack(side="left", padx=5)
        self.file_label = ttk.Label(top, text="Nog geen verbruiks-CSV geladen")
        self.file_label.pack(side="left", padx=10)

        ttk.Button(top, text="Prijs-CSV kiezen", command=self.load_price_csv).pack(side="left", padx=5)
        self.price_file_label = ttk.Label(top, text="Geen prijs-CSV geladen")
        self.price_file_label.pack(side="left", padx=10)

        ttk.Button(top, text="ENTSO-E prijzen ophalen", command=self.download_entsoe_prices).pack(side="left", padx=5)

        settings = ttk.LabelFrame(self.root, text="Instellingen", padding=10)
        settings.pack(fill="x", padx=10, pady=5)
        ttk.Label(settings, text="Afnameprijs €/kWh").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.import_price_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(settings, text="Terugleververgoeding €/kWh").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.export_price_var, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(settings, text="Start SoC %").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.initial_soc_var, width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(settings, text="Frank opslag €/kWh").grid(row=0, column=6, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.frank_opslag_var, width=10).grid(row=0, column=7, sticky="w")
        ttk.Label(settings, text="Prijsmode").grid(row=0, column=8, sticky="w", padx=5, pady=5)
        ttk.Combobox(settings, textvariable=self.price_mode_var, values=["fixed", "dynamic_csv", "entsoe_api"], state="readonly", width=14).grid(row=0, column=9, sticky="w")
        ttk.Label(settings, text="ENTSO-E token").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.entsoe_token_var, width=35).grid(row=1, column=1, columnspan=3, sticky="w")
        ttk.Label(settings, text="Bidding zone").grid(row=1, column=4, sticky="w", padx=5, pady=5)
        ttk.Entry(settings, textvariable=self.entsoe_zone_var, width=20).grid(row=1, column=5, sticky="w")

        battery_frame = ttk.LabelFrame(self.root, text="Batterijen", padding=10)
        battery_frame.pack(fill="x", padx=10, pady=5)
        self.battery_listbox = tk.Listbox(battery_frame, selectmode=tk.MULTIPLE, height=7, exportselection=False)
        self.battery_listbox.pack(side="left", fill="x", expand=True)
        self.refresh_battery_list()
        button_col = ttk.Frame(battery_frame)
        button_col.pack(side="left", padx=10)
        ttk.Button(button_col, text="Batterij toevoegen", command=self.open_add_battery_window).pack(pady=5, fill="x")
        ttk.Button(button_col, text="Batterij wijzigen", command=self.open_edit_battery_window).pack(pady=5, fill="x")
        ttk.Button(button_col, text="Batterij verwijderen", command=self.delete_selected_battery).pack(pady=5, fill="x")
        ttk.Button(button_col, text="Simulatie uitvoeren", command=self.run_simulation).pack(pady=5, fill="x")
        ttk.Button(button_col, text="Samenvatting opslaan", command=self.save_summary).pack(pady=5, fill="x")

        lower = ttk.Frame(self.root)
        lower.pack(fill="both", expand=True, padx=10, pady=10)
        left = ttk.Frame(lower)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(lower)
        right.pack(side="left", fill="both", expand=True)

        info_frame = ttk.LabelFrame(left, text="Algemene resultaten", padding=8)
        info_frame.pack(fill="x", pady=(0, 8))
        self.overall_text = tk.Text(info_frame, height=8, wrap="word")
        self.overall_text.pack(fill="x")

        ttk.Label(left, text="Samenvatting per batterij").pack(anchor="w")
        self.summary_tree = ttk.Treeview(left, show="headings", height=16)
        self.summary_tree.pack(fill="both", expand=True)

        detail_top = ttk.Frame(right)
        detail_top.pack(fill="x")
        ttk.Label(detail_top, text="Detailbatterij").pack(side="left")
        self.detail_var = tk.StringVar()
        self.detail_combo = ttk.Combobox(detail_top, textvariable=self.detail_var, state="readonly", width=40)
        self.detail_combo.pack(side="left", padx=5)
        ttk.Button(detail_top, text="Toon grafieken", command=self.draw_selected_battery).pack(side="left", padx=5)

        self.metrics_text = tk.Text(right, height=14, wrap="word")
        self.metrics_text.pack(fill="x", pady=5)
        self.fig = plt.Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_battery_list(self):
        current_selection = set(self.battery_listbox.curselection())
        self.battery_listbox.delete(0, tk.END)
        for battery in self.all_batteries:
            self.battery_listbox.insert(tk.END, battery.name)
        if current_selection:
            for i in current_selection:
                if i < len(self.all_batteries):
                    self.battery_listbox.selection_set(i)
        else:
            for i in range(len(self.all_batteries)):
                self.battery_listbox.selection_set(i)

    def _open_battery_window(self, edit_index: Optional[int]):
        win = tk.Toplevel(self.root)
        win.title("Batterij aanpassen" if edit_index is not None else "Batterij toevoegen")
        win.geometry("430x290")
        base = self.all_batteries[edit_index] if edit_index is not None else BatterySpec("Eigen batterij", 10.0, 5.0, 5.0, 0.92, 6000)
        fields = {
            "Naam": tk.StringVar(value=base.name),
            "Bruikbare capaciteit (kWh)": tk.DoubleVar(value=base.usable_kwh),
            "Max laadvermogen (kW)": tk.DoubleVar(value=base.max_charge_kw),
            "Max ontlaadvermogen (kW)": tk.DoubleVar(value=base.max_discharge_kw),
            "Roundtrip efficiency": tk.DoubleVar(value=base.roundtrip_efficiency),
            "Aanschafprijs (€)": tk.DoubleVar(value=base.purchase_price_eur),
        }
        for i, (label, var) in enumerate(fields.items()):
            ttk.Label(win, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=6)
            ttk.Entry(win, textvariable=var, width=25).grid(row=i, column=1, padx=10, pady=6)

        def save_battery():
            battery = BatterySpec(
                name=fields["Naam"].get(),
                usable_kwh=float(fields["Bruikbare capaciteit (kWh)"].get()),
                max_charge_kw=float(fields["Max laadvermogen (kW)"].get()),
                max_discharge_kw=float(fields["Max ontlaadvermogen (kW)"].get()),
                roundtrip_efficiency=float(fields["Roundtrip efficiency"].get()),
                purchase_price_eur=float(fields["Aanschafprijs (€)"].get()),
            )
            if edit_index is None:
                self.all_batteries.append(battery)
            else:
                self.all_batteries[edit_index] = battery
            self.refresh_battery_list()
            win.destroy()

        ttk.Button(win, text="Opslaan", command=save_battery).grid(row=len(fields), column=0, columnspan=2, pady=10)

    def open_add_battery_window(self):
        self._open_battery_window(None)

    def open_edit_battery_window(self):
        selected = self.battery_listbox.curselection()
        if len(selected) != 1:
            messagebox.showwarning("Let op", "Selecteer precies 1 batterij om te wijzigen.")
            return
        self._open_battery_window(selected[0])

    def delete_selected_battery(self):
        selected = list(self.battery_listbox.curselection())
        if len(selected) != 1:
            messagebox.showwarning("Let op", "Selecteer precies 1 batterij om te verwijderen.")
            return
        index = selected[0]
        name = self.all_batteries[index].name
        if messagebox.askyesno("Bevestigen", f"Batterij '{name}' verwijderen?"):
            del self.all_batteries[index]
            self.refresh_battery_list()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.df = load_energy_csv(path)
            self.file_label.config(text=path)
            if self.df.attrs.get("data_mode") == "homewizard_p1":
                messagebox.showinfo("Gelukt", f"HomeWizard P1 CSV geladen: {len(self.df)} kwartierregels.\nDeze export bevat netmeterstanden (import/export).\nBesparing en laad/ontlaadprofiel zijn bruikbaar; zelfconsumptie/autarkie zijn indicatief of n.v.t.")
            else:
                messagebox.showinfo("Gelukt", f"CSV geladen: {len(self.df)} regels")
        except Exception as exc:
            messagebox.showerror("CSV fout", str(exc))

    def load_price_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.price_df = load_price_csv(path)
            self.price_file_label.config(text=path)
            messagebox.showinfo("Gelukt", f"Prijs-CSV geladen: {len(self.price_df)} prijsregels")
        except Exception as exc:
            messagebox.showerror("Prijs-CSV fout", str(exc))

    def download_entsoe_prices(self):
        if self.df is None:
            messagebox.showwarning("Let op", "Laad eerst je verbruiks-CSV zodat de periode bekend is.")
            return
        token = self.entsoe_token_var.get().strip()
        if not token:
            messagebox.showwarning("Let op", "Vul eerst je ENTSO-E token in.")
            return
        try:
            self.price_df = download_entsoe_prices_for_period(self.df, self.entsoe_zone_var.get().strip(), token)
            self.price_file_label.config(text=f"ENTSO-E prijzen geladen: {len(self.price_df)} regels")
            self.price_mode_var.set("entsoe_api")
            messagebox.showinfo("Gelukt", f"ENTSO-E day-ahead prijzen opgehaald: {len(self.price_df)} regels.\nFrank opslag wordt apart op de afnameprijs gezet.")
        except Exception as exc:
            messagebox.showerror("ENTSO-E fout", str(exc))

    def run_simulation(self):
        if self.df is None:
            messagebox.showwarning("Let op", "Laad eerst een CSV.")
            return
        selected_indices = self.battery_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Let op", "Selecteer minimaal 1 batterij.")
            return
        try:
            aligned_df = align_prices_to_energy(
                self.df,
                self.price_df,
                self.price_mode_var.get(),
                float(self.import_price_var.get()),
                float(self.export_price_var.get()),
                float(self.frank_opslag_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Prijsfout", str(exc))
            return

        summary_rows = []
        self.simulations.clear()
        self.metrics_per_battery.clear()
        self.financials_per_battery.clear()
        for i in selected_indices:
            battery = self.all_batteries[i]
            sim_df, metrics = simulate_battery(aligned_df, battery, float(self.initial_soc_var.get()) / 100.0)
            financials = calculate_financials(sim_df, metrics, battery.purchase_price_eur)
            self.simulations[battery.name] = sim_df
            self.metrics_per_battery[battery.name] = metrics
            self.financials_per_battery[battery.name] = financials
            summary_rows.append({
                "Batterij": battery.name,
                "Capaciteit (kWh)": round(battery.usable_kwh, 2),
                "Laadvermogen (kW)": round(battery.max_charge_kw, 2),
                "Ontlaadvermogen (kW)": round(battery.max_discharge_kw, 2),
                "Lading gebruikt (kWh)": round(metrics["battery_charge_kwh"], 1),
                "Ontlading gebruikt (kWh)": round(metrics["battery_discharge_kwh"], 1),
                "Netafname met batterij (kWh)": round(metrics["grid_import_with_battery_kwh"], 1),
                "Teruglevering met batterij (kWh)": round(metrics["grid_export_with_battery_kwh"], 1),
                "Kosten zonder batterij (€)": round(financials["cost_without_battery_eur"], 2),
                "Kosten met batterij (€)": round(financials["cost_with_battery_eur"], 2),
                "Theoretische besparing (€)": round(financials["theoretical_saving_eur"], 2),
                "Aanschafprijs (€)": round(battery.purchase_price_eur, 2),
                "Terugverdientijd (jaar)": round(financials["simple_payback_years"], 1) if pd.notna(financials["simple_payback_years"]) else np.nan,
            })
        self.summary_df = pd.DataFrame(summary_rows).sort_values("Theoretische besparing (€)", ascending=False)
        self.populate_summary_tree()
        self.populate_overall_text(aligned_df)
        names = list(self.simulations.keys())
        self.detail_combo["values"] = names
        if names:
            self.detail_var.set(names[0])
            self.draw_selected_battery()

    def populate_overall_text(self, aligned_df: pd.DataFrame):
        self.overall_text.delete("1.0", tk.END)
        self.overall_text.insert(tk.END, f"Periode: {aligned_df['timestamp'].min()} t/m {aligned_df['timestamp'].max()}\n")
        self.overall_text.insert(tk.END, f"Netafname zonder batterij: {aligned_df['deficit_kwh'].sum():.1f} kWh\n")
        self.overall_text.insert(tk.END, f"Teruglevering zonder batterij: {aligned_df['surplus_kwh'].sum():.1f} kWh\n")
        self.overall_text.insert(tk.END, f"Prijsmode: {aligned_df.attrs.get('price_mode', 'fixed')}\n")
        self.overall_text.insert(tk.END, f"Frank opslag: {aligned_df.attrs.get('frank_opslag', 0):.3f} €/kWh\n")
        if aligned_df.attrs.get("price_mode") in {"dynamic_csv", "entsoe_api"}:
            self.overall_text.insert(tk.END, f"Prijs-overlap met energiedata: {aligned_df.attrs.get('price_overlap_ratio', 1.0) * 100:.1f}%\n")
            self.overall_text.insert(tk.END, "Niet-overlappende regels vallen terug op de vaste handmatige prijzen.\n")
        if self.df is not None and self.df.attrs.get("data_mode") == "homewizard_p1":
            self.overall_text.insert(tk.END, "HomeWizard P1 bevat netmeterstanden; zelfconsumptie/autarkie blijven indicatief.\n")

    def populate_summary_tree(self):
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        if self.summary_df is None or self.summary_df.empty:
            return
        columns = list(self.summary_df.columns)
        self.summary_tree["columns"] = columns
        for col in columns:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=125, anchor="center")
        for _, row in self.summary_df.iterrows():
            self.summary_tree.insert("", "end", values=[row[col] for col in columns])

    def draw_selected_battery(self):
        name = self.detail_var.get()
        if not name or name not in self.simulations:
            return
        df = self.simulations[name]
        metrics = self.metrics_per_battery[name]
        financials = self.financials_per_battery[name]
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, f"Batterij: {name}\n")
        self.metrics_text.insert(tk.END, f"Theoretische besparing: € {financials['theoretical_saving_eur']:.2f}\n")
        self.metrics_text.insert(tk.END, f"Kosten zonder batterij: € {financials['cost_without_battery_eur']:.2f}\n")
        self.metrics_text.insert(tk.END, f"Kosten met batterij: € {financials['cost_with_battery_eur']:.2f}\n")
        self.metrics_text.insert(tk.END, f"Netafname zonder batterij: {metrics['grid_import_without_battery_kwh']:.1f} kWh\n")
        self.metrics_text.insert(tk.END, f"Netafname met batterij: {metrics['grid_import_with_battery_kwh']:.1f} kWh\n")
        self.metrics_text.insert(tk.END, f"Teruglevering zonder batterij: {metrics['grid_export_without_battery_kwh']:.1f} kWh\n")
        self.metrics_text.insert(tk.END, f"Teruglevering met batterij: {metrics['grid_export_with_battery_kwh']:.1f} kWh\n")

        self.fig.clear()
        ax1 = self.fig.add_subplot(411)
        ax2 = self.fig.add_subplot(412)
        ax3 = self.fig.add_subplot(413)
        ax4 = self.fig.add_subplot(414)
        show_df = df.tail(96 * 3)
        x = show_df["timestamp"]
        ax1.plot(x, show_df["consumption_kwh"], label="Verbruik")
        ax1.plot(x, show_df["production_kwh"], label="Opwek")
        ax1.set_title("Verbruik vs Opwek")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(x, show_df["soc_kwh"], label="SoC batterij")
        ax2.set_title("State of Charge")
        ax2.set_ylabel("kWh")
        ax2.grid(True)
        ax2.legend()
        labels = ["Zonder batterij", "Met batterij"]
        imports = [metrics["grid_import_without_battery_kwh"], metrics["grid_import_with_battery_kwh"]]
        exports = [metrics["grid_export_without_battery_kwh"], metrics["grid_export_with_battery_kwh"]]
        idx = np.arange(len(labels))
        width = 0.35
        ax3.bar(idx - width / 2, imports, width, label="Netafname")
        ax3.bar(idx + width / 2, exports, width, label="Teruglevering")
        ax3.set_xticks(idx)
        ax3.set_xticklabels(labels)
        ax3.set_title("Netvergelijking")
        ax3.set_ylabel("kWh")
        ax3.legend()
        ax3.grid(True, axis="y")
        if "import_price_eur_per_kwh" in show_df.columns:
            ax4.plot(x, show_df["import_price_eur_per_kwh"], label="Afnameprijs")
            if "export_price_eur_per_kwh" in show_df.columns:
                ax4.plot(x, show_df["export_price_eur_per_kwh"], label="Terugleverprijs")
            ax4.set_title("Prijsprofiel")
            ax4.set_ylabel("€/kWh")
            ax4.grid(True)
            ax4.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def save_summary(self):
        if self.summary_df is None or self.summary_df.empty:
            messagebox.showwarning("Let op", "Voer eerst een simulatie uit.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        self.summary_df.to_csv(path, index=False)
        messagebox.showinfo("Opgeslagen", f"Samenvatting opgeslagen naar:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BatterySimulatorApp(root)
    root.mainloop()
