export default class RegionGeoMixin {
  static getGeoURL() {
    const classId = this.regionClassId();
    return (
      `https://raw.githubusercontent.com` +
      `/nuuuwan/lk_admin_regions` +
      `/refs/heads/main` +
      `/data/geo/topojson/e4_medium/` +
      `${classId}.topojson`
    );
  }
}
