import matplotlib.pyplot as plt

from lanka_data.where.Regions import log


class RegionsMapMixin:
    MAX_REGIONS_TO_LABEL = 100

    def draw_map(self, file_path_base: str):
        gdf_region = self._get_geopandas_dataframe()
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

        if n_regions <= self.MAX_REGIONS_TO_LABEL:
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
        }
