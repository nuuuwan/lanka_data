import Concept from "../../Concept.js";

export default class Party extends Concept {
  static fromValue(value) {
    return new this(value);
  }

  getLabel() {
    const label = super.getLabel();
    return label === this.value ? label.toUpperCase() : label;
  }

  static getColorMap() {
    return {
      SLFP: "#00058f",
      PA: "#00058f",
      UPFA: "#00058f",
      UNP: "#00b10c",
      NDF: "#00b10c",
      IND16: "#00b10c",
      SJB: "#88cc00",
      SLPP: "#9e1420",
      OPPP: "#880000",
      SLMP: "#880088",
      ACMC: "#004400",
      MNA: "#004400",
      NC: "#004400",
      SLMC: "#004400",
      NUA: "#004400",
      IG: "#e0e0e0",
      IG2: "#e0e0e0",
      IG3: "#e0e0e0",
      DUNF: "#8800ff",
      SB: "#0088ff",
      JVP: "#ff0000",
      NMPP: "#ff0000",
      NPP: "#ff0000",
      NPPT: "#ff0000",
      MEP: "#ff0000",
      USA: "#ff0000",
      SLPF: "#ff0000",
      DNA: "#ff0000",
      JJB: "#ff0000",
      LSSP: "#ff0000",
      CP: "#ff0000",
      NSSP: "#ff0000",
      FSP: "#ff0000",
      SEP: "#ff0000",
      ELMSP: "#ff6200",
      EPDP: "#ff6200",
      TMVP: "#ff6200",
      EROS: "#ff6200",
      INDI: "#ff6200",
      CWC: "#ff4400",
      UPF: "#ff4400",
      SU: "#ffcc00",
      JHU: "#ffcc00",
      AITC: "#ffdd00",
      ITAK: "#ffdd00",
      TULF: "#ffdd00",
      ACTC: "#ffdd00",
      TMK: "#ffdd00",
      TMTK: "#ffdd00",
      IND9: "#ffdd00",
      ELJP: "#ffdd00",
    };
  }

  getColor() {
    return this.constructor.getColorMap()[this.value] ?? super.getColor();
  }
}
