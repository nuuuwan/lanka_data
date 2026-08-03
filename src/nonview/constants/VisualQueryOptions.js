export const DIMENSION_OPERATORS = [
  { value: "", label: "None" },
  { value: "=", label: "=" },
  { value: ":", label: ":" },
  { value: "<", label: "<" },
];

export const FIELD_GROUPS = [
  {
    label: "Time",
    fields: ["Time", "TimeDurationGroup", "TimeGroup0510More"],
  },
  {
    label: "Geography",
    fields: ["Country", "Province", "District", "DSD", "ED", "GND", "PD", "Region"],
  },
  {
    label: "Demographics",
    fields: [
      "AgeGroup",
      "AgeGroupWorking",
      "DisabilityTypes",
      "EconomicInactivityReason",
      "EducationActivity",
      "EmmigrationReason",
      "EmploymentStatus",
      "Ethnicity",
      "HighestEducationLevel",
      "HighestEducationLevel2",
      "HighestEducationLevel3",
      "IsEconomicallyActive",
      "LanguageLiteracy",
      "LiveBirths",
      "MaritalStatus",
      "MigrationDirection",
      "MigrationLifetimeDirection",
      "MigrationReason",
      "MigrationStatus",
      "NonCommunicableDisease",
      "Religion",
      "ResidentRelativeToDistrict",
      "Sex",
      "SingleOrMultipleDisabilities",
    ],
  },
  {
    label: "Households",
    fields: [
      "CookingFuel",
      "FloorType",
      "HouseholdAppliances",
      "HouseholdOccupancy",
      "HouseholdSize",
      "HouseholdStructure",
      "HouseholdType",
      "Lighting",
      "LiquidWasteDisposal",
      "LivingQuarters",
      "OccupationStatus",
      "OneRoomOrMore",
      "OwnershipStatus",
      "RoofType",
      "SolidWasteDisposal",
      "SourceOfDrinkingWater",
      "ToiletFacilities",
      "TypeOfUnit",
      "WallType",
      "WaterSupplyAvailability",
    ],
  },
  {
    label: "Census",
    fields: ["Census", "CensusOfficer", "CensusTopic"],
  },
  {
    label: "Government",
    fields: ["AdministrativeEntity", "Sector"],
  },
  {
    label: "Elections",
    fields: ["ElectionType", "Party", "Summary"],
  },
];

export const VISUAL_GROUPS = [
  {
    label: "Charts",
    visuals: [
      "AreaBump",
      "BarChart",
      "StackedBarChart",
      "MarimekkoChart",
      "PieChart",
      "TreeMap",
    ],
  },
  {
    label: "Maps",
    visuals: [
      "Map",
      "Cartogram",
      "HexMap",
      "UnitHexMap",
      "SquareMap",
      "UnitSquareMap",
    ],
  },
  {
    label: "Other",
    visuals: ["Blocks", "Table", "JSON"],
  },
];
