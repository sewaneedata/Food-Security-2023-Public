from __future__ import annotations
import pathlib
import pandas as pd
import shiny.experimental as x
from shiny import App, Inputs, Outputs, Session, render, ui
from htmltools import css
from PIL import Image


dir = pathlib.Path(__file__).parent

app_ui = ui.page_fluid(
    ui.panel_title("", "DataLab: Feeding Futures"),
    ui.row(
        x.ui.card_header("DataLab", style="font-size: 36px; color: white; background-color: rgba(95, 54, 110); text-align: left"),
        ui.navset_tab(
            ui.nav("Home", x.ui.card(
                x.ui.card_header("Feeding Futures", class_="text-center", style="font-size: 24px; color: white; background-color: rgba(95, 54, 110)"),
                ui.row(
                    ui.column(4, ui.p(ui.strong("Food Bank Sites 2023:"), style="text-align:center; font-size:20px")),
                    ui.column(4, ui.p(ui.strong("110,000 meals served since 2015"), style="text-align:center; font-size:40px")),
                    ui.column(4, ui.p(ui.strong("Meal Sites 2023:"), style="text-align:center; font-size:20px")),
                ),
                # Add tables here
                ui.row(
                    ui.column(6, ui.output_table("food_bank_table")),
                    ui.column(6, ui.output_table("meal_sites_table")),
                ),
                x.ui.card_image(
                    file=None,
                    src="",
                ),
                x.ui.card_footer("", class_="text-center", style="background-color: rgba(95, 54, 110); color: white; font-size: 18px"),
                full_screen=True,
            )),
        ui.nav("About Us", 
               ui.row(
                   ui.column(6, "Over 34 million people face food insecurity in the US, including 1 in 8 children. In the South Cumberland region in Middle Tennessee, the food insecurity ratio for kids is even more alarming: 1 in 3. The South Cumberland Summer Meal Program (SCSMP) aims to fight hunger during the most vulnerable time for kids - in the summer, when schools don’t provide lunches - and has served over 110,000 meals since 2015. One of the challenges the program faces is the difficulty in predicting accurately ahead of time the quantity of meals needed every week for every site. As a result, a significant percent of meals ordered go to waste and are also not reimbursed by the USDA, which has a negative impact on the funds available for the following year’s program. Here is the link to volunteer as an AmeriCorps VISTA for the south Cumberland Plaeteu: ", ui.a("https://my.americorps.gov/mp/listing/viewListing.do?id=64500&fromSearch=true", href = "https://my.americorps.gov/mp/listing/viewListing.do?id=64500&fromSearch=true")),
                           ui.column(6,
            ui.p("To address this problem, the data science project analyzed the data, gathered insights, and developed a versatile predictive tool using historical data, weather conditions, site locations, population, and local events to enable more accurate meal orders and efficient operations for the program. This will contribute towards reducing the number of food insecure families in the areas SCSMP serves, and also ultimately to the UN’s Sustainable Development Goal: Zero Hunger. For more information, please visit our homepage: ", ui.a("https://new.sewanee.edu/sewanee-datalab/", href = "https://new.sewanee.edu/sewanee-datalab/"), " and take at look at our github for the open source code for the dashboard: ", ui.a("https://github.com/sewaneedata/Food-Security-2023-Public", href = "https://github.com/sewaneedata/Food-Security-2023-Public"), style="text-align:center"),
        )
    ),
    ui.row(
        ui.column(6,
            ui.div(
                {"style": "display: inline-block"},
                ui.img(src = "scsmp.png", height = "120em"),
                ui.img(src = "usda.png", height = "120em"),
                ui.img(src = "americorps.png", height = "120em"),
                ui.img(src = "no-kid-hungry.png", height = "120em"),
                ui.img(src = "datalab.png", height = "120em"),
            ),
        ),
        ui.column(6,
            ui.div(
                {"style": "display: inline-block"},
                ui.tags.figure(
                    {"style": "display: inline-block"},
                    ui.img(src = "Carson.jpg", height = "120em"),
                    ui.tags.figcaption("Carson Coody"),
                ),
                ui.tags.figure(
                    {"style": "display: inline-block"},
                    ui.img(src = "Eden.jpg", height = "120em"),
                    ui.tags.figcaption("Eden Balem"),
                ),
                ui.tags.figure(
                    {"style": "display: inline-block"},
                    ui.img(src = "Orion.jpg", height = "120em"),
                    ui.tags.figcaption("Orion Gant"),
                ),
                ui.tags.figure(
                    {"style": "display: inline-block"},
                    ui.img(src = "Bryce.jpg", height = "120em"),
                    ui.tags.figcaption("Bryce Comer"),
                ),
            ),
        )),
    ))
))

def server(input: Inputs, output: Outputs, session: Session):
    # Read data from CSV files
    food_bank_data = pd.read_csv(dir / "2023_bank_schedules.csv")
    meal_sites_data = pd.read_csv(dir / "2023_flyer_schedules.csv")

    # img_path = dir / "SCSMP.png"

    @output
    @render.table(classes='table shiny-table w-50')
    def food_bank_table():
        # Return food bank data
        return food_bank_data

    @output
    @render.table(classes='table shiny-table w-50')
    def meal_sites_table():
        # Return meal sites data
        return meal_sites_data

app = App(app_ui, server, static_assets = dir)
