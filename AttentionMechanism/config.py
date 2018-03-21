class Config:
    def __init__(self, **kwargs):
        # path
        self.glove_path = kwargs["glove_path"]
        self.event_data_path = kwargs["event_data_path"]

        # embedding
        self.embedding_size = kwargs.get("embedding_size", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # data
        self.context_window_size = kwargs["context_window_size"]
        self.entity_dim = kwargs["entity_dim"]
        self.entity_size = kwargs.get("entity_size", None)
        self.output_size = kwargs.get("output_size", None)

        # training setting
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.lamb = kwargs["lamb"]
        self.drop_rate = kwargs["drop_rate"]
        self.machine = kwargs["machine"]
        self.negative_ratio = kwargs["negative_ratio"]
        self.remove_list = kwargs["remove_list"]
        self.trigger_dictionary = kwargs["trigger_dictionary"]
        self.reinforce_list = kwargs["reinforce_list"]

entity_type_dictionary = {
    "O":0,
    "DATE":1,
    "PERSON":2,
    "ORGANIZATION":3
}

trigger_type_dictionary = {
    "none":0,
    "business.employment_tenure.trigger":1,
    "business.sponsorship.trigger":2,
    "education.education.trigger":3,
    "film.dubbing_performance.trigger":4,
    "film.film_crew_gig.trigger":5,
    "military.military_command.trigger":6,
    "military.military_service.trigger":7,
    "music.group_membership.trigger":8,
    "music.track_contribution.trigger":9,
    "olympics.olympic_athlete_affiliation.trigger":10,
    "olympics.olympic_medal_honor.trigger":11,
    "organization.leadership.trigger":12,
    "organization.organization_board_membership.trigger":13,
    "people.appointment.trigger":14,
    "people.marriage.trigger":15,
    "people.place_lived.trigger":16,
    "projects.project_participation.trigger":17,
    "sports.sports_team_roster.trigger":18,
    "sports.sports_team_season_record.trigger":19,
    "wine.grape_variety_composition.trigger":20,
}

binary_trigger_type_dictionary = {
    "none":0,
    "business.employment_tenure.trigger":1,
    "business.sponsorship.trigger":1,
    "education.education.trigger":1,
    "film.dubbing_performance.trigger":1,
    "film.film_crew_gig.trigger":1,
    "military.military_command.trigger":1,
    "military.military_service.trigger":1,
    "music.group_membership.trigger":1,
    "music.track_contribution.trigger":1,
    "olympics.olympic_athlete_affiliation.trigger":1,
    "olympics.olympic_medal_honor.trigger":1,
    "organization.leadership.trigger":1,
    "organization.organization_board_membership.trigger":1,
    "people.appointment.trigger":1,
    "people.marriage.trigger":1,
    "people.place_lived.trigger":1,
    "projects.project_participation.trigger":1,
    "sports.sports_team_roster.trigger":1,
    "sports.sports_team_season_record.trigger":1,
    "wine.grape_variety_composition.trigger":1,
}

remove_list_less_50 = [
    "film.dubbing_performance.trigger",
    "music.track_contribution.trigger",
    "projects.project_participation.trigger",
    "wine.grape_variety_composition.trigger",
]

remove_list_less_100 = [
    "film.dubbing_performance.trigger",
    "music.track_contribution.trigger",
    "projects.project_participation.trigger",
    "wine.grape_variety_composition.trigger",
    "business.sponsorship.trigger",
    "people.appointment.trigger",
]

reinforce_list = [
    "business.employment_tenure.trigger",
    "film.film_crew_gig.trigger",
    "military.military_command.trigger",
    "olympics.olympic_athlete_affiliation.trigger",
    "olympics.olympic_medal_honor.trigger",
    "organization.organization_board_membership.trigger",
    "people.place_lived.trigger",
    "sports.sports_team_roster.trigger",
    "sports.sports_team_season_record.trigger",
]
