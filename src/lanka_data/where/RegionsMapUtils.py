import matplotlib.pyplot as plt

from lanka_data.where.RegionsGeoUtils import RegionsGeoUtils
from utils_future import Log

log = Log("RegionsMapUtils")


class RegionsMapUtils:
    MAX_REGIONS_TO_LABEL = 100

    @staticmethod
    def draw_map(result, file_path_base: str):
        data_list = result["data_list"]
        region_ids = [d["region_id"] for d in data_list]
        gdf_region = RegionsGeoUtils.get_geopandas_dataframe(region_ids)
        n_regions = len(gdf_region)
        cmap = plt.cm.tab20  # pylint: disable=no-member.
        colors = [cmap(i % 20) for i in range(n_regions)]
        gdf_region = gdf_region.copy()
        gdf_region["color"] = colors

        fig, ax = plt.subplots(figsize=(10, 8))
        gdf_region.plot(
            ax=ax,
            categorical=True,
            color=gdf_region["color"],
            edgecolor="white",
            linewidth=0.2,
        )

        if n_regions <= RegionsMapUtils.MAX_REGIONS_TO_LABEL:
            for _, row in gdf_region.iterrows():
                centroid = row.geometry.centroid
                ax.annotate(
                    row["id"],
                    xy=(centroid.x, centroid.y),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )
        ax.set_axis_off()

        image_path = f"{file_path_base}.png"
        fig.savefig(image_path, dpi=200, bbox_inches="tight")
        log.info(f"Wrote {image_path}")
        plt.close(fig)
        return {
            "image_path": image_path,
            "source": "Department of Census and Statistics, Sri Lanka",
            "source_url": "https://www.statistics.gov.lk/",
        }
