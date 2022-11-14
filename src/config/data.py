from os.path import abspath, dirname
from pydantic import BaseModel, Field


class DatasetsConfig(BaseModel):
    main_dir = dirname(dirname(dirname(abspath(__file__))))
    data_dir = f'{main_dir}/datasets/timeseries'
    daily_temperatures: str = Field(f'{data_dir}/daily-minimum-temperatures-in-me.csv')
    electricity_production: str = Field(f'{data_dir}/Electric_Production.csv')
    beer_production: str = Field(f'{data_dir}/monthly-beer-production-in-austr.csv')
    shampoo_sales: str = Field(f'{data_dir}/sales-of-shampoo-over-a-three-ye.csv')